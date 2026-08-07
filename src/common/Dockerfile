FROM node:23 AS common-builder

WORKDIR /app

RUN npm install -g pnpm

COPY pnpm-workspace.yaml pnpm-lock.yaml package.json ./

# Copy ALL workspace package.json files so pnpm resolves the full workspace.
# Without these, `pnpm install` strips missing workspaces from the lockfile,
# which then breaks the app frontend-builder stage (--frozen-lockfile fails).
COPY frontend/package.json frontend/

WORKDIR /app/common
COPY common/package.json common/tsconfig.json ./
COPY common/src ./src

RUN pnpm install
