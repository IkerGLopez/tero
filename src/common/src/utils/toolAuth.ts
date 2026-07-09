export enum ToolAuthType {
  OAUTH = 'oauth',
  TOKEN = 'token'
}

export class OAuthRequest {
  toolId: string;
  url: string;
  state: string;
  agentId: number;

  constructor(toolId: string, url: string, state: string, agentId: number) {
    this.toolId = toolId;
    this.url = url;
    this.state = state;
    this.agentId = agentId;
  }
}

export class AuthTokenRequest {
  toolId: string;
  agentId: number;

  constructor(toolId: string, agentId: number) {
    this.toolId = toolId;
    this.agentId = agentId;
  }
}

export class AuthenticationError extends Error {
  errorCode: string;

  constructor(errorCode: string) {
    super('Authentication error: ' + errorCode);
    this.errorCode = errorCode;
  }
}

export interface HttpErrorLike {
  status: number;
  body?: string;
}

export interface AuthTokenRequestData {
  toolId: string;
  agentId: number;
}

export interface AuthTokenResult {
  success: boolean;
  token?: string;
  cancelled?: boolean;
}

export interface ToolAuthCallbacks {
  onShowOAuth: (toolId: string, onCancel: () => void) => void;
  onHideOAuth: () => void;
  onRequestToken: (request: AuthTokenRequestData) => Promise<AuthTokenResult>;
}

export class ToolAuthManager {
  private handler: ToolAuthCallbacks | null = null;

  register(handler: ToolAuthCallbacks): void {
    this.handler = handler;
  }

  unregister(): void {
    this.handler = null;
  }

  async requestToken(request: AuthTokenRequestData): Promise<AuthTokenResult> {
    if (!this.handler) {
      return { success: false, cancelled: true };
    }
    return this.handler.onRequestToken(request);
  }

  showOAuthRequired(toolId: string, onCancel: () => void): void {
    this.handler?.onShowOAuth(toolId, onCancel);
  }

  hideOAuthRequired(): void {
    this.handler?.onHideOAuth();
  }
}

export const toolAuthManager = new ToolAuthManager();

export interface AuthFlowHandlers {
  isHttpError: (e: unknown) => e is HttpErrorLike;
  handleOAuthFlow: (request: OAuthRequest) => Promise<void>;
  completeAuthToken?: (toolId: string, agentId: number, token: string) => Promise<void>;
}

export const createAuthFlowHandler = (handlers: AuthFlowHandlers) => {
  const authTokenFlow = async (request: AuthTokenRequest): Promise<void> => {
    const result = await toolAuthManager.requestToken({
      toolId: request.toolId,
      agentId: request.agentId,
    });

    if (!result.success || !result.token) {
      throw new AuthenticationError(result.cancelled ? 'authenticationCancelled' : 'authenticationAccessDenied');
    }

    await handlers.completeAuthToken!(request.toolId, request.agentId, result.token);
  };

  return async <T>(fn: () => Promise<T>): Promise<T> => {
    while (true) {
      try {
        return await fn();
      } catch (e) {
        if (!handlers.isHttpError(e) || e.status !== 401 || !e.body) {
          throw e;
        }
        const authRequest = parseAuthRequest(e);
        if (authRequest) {
          if (authRequest instanceof OAuthRequest) {
            await handlers.handleOAuthFlow(authRequest);
          } else if (authRequest instanceof AuthTokenRequest) {
            if (!handlers.completeAuthToken) {
              throw e;
            }
            await authTokenFlow(authRequest);
          }
          continue;
        }
        throw e;
      }
    }
  };
};

export const parseAuthRequest = (
  e: any,
): OAuthRequest | AuthTokenRequest | undefined => {
  try {
    const body = JSON.parse(e.body);
    if (!body.detail) {
      return;
    }
    if (body.detail.requestType === "oauth") {
      return new OAuthRequest(body.detail.toolId, body.detail.oauthUrl, body.detail.oauthState, body.detail.agentId);
    }
    else if (body.detail.requestType === "auth_token") {
      return new AuthTokenRequest(body.detail.toolId, body.detail.agentId);
    }
    return;
  } catch (_) {
    return;
  }
};
