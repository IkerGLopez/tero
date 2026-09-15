export const RETRIEVED_PREVIEW_LENGTH = 150

// Retrieved chunks carry full content end to end; previews are truncated here,
// at the presentation layer, so the payload and the eval channel stay lossless.
export const truncatePreview = (text: string, maxLength: number = RETRIEVED_PREVIEW_LENGTH): string => {
  return text.length > maxLength ? text.slice(0, maxLength) + '...' : text
}
