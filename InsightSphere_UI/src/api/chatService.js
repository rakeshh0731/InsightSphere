const API_BASE_URL = ''; // Using proxy, so no base URL needed here

/**
 * Starts a new chat session.
 * @returns {Promise<string>} The new session ID.
 */
export const startSession = async () => {
  const response = await fetch(`${API_BASE_URL}/session/start`, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
  });
  if (!response.ok) {
    throw new Error(`API Error: ${response.status} ${response.statusText}`);
  }
  const data = await response.json();
  return data.chatSessionId;
};

/**
 * Sends a message to the chat API.
 * @param {string} chatSessionId - The current session ID.
 * @param {string} message - The user's message.
 * @returns {Promise<string>} The bot's reply.
 */
export const sendMessage = async (chatSessionId, message) => {
  const response = await fetch(`${API_BASE_URL}/chat`, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({ chatSessionId, message }),
  });
  if (!response.ok) {
    throw new Error(`API Error: ${response.status} ${response.statusText}`);
  }
  const data = await response.json();
  return data.reply;
};

/**
 * Clears the history of a chat session.
 * @param {string} chatSessionId - The current session ID.
 * @returns {Promise<object>} The API response.
 */
export const resetSession = async (chatSessionId) => {
  const response = await fetch(`${API_BASE_URL}/session/clear`, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({ chatSessionId }),
  });
  if (!response.ok) {
    throw new Error(`API Error: ${response.status} ${response.statusText}`);
  }
  return response.json();
};

/**
 * Closes a session. Uses navigator.sendBeacon for reliability on page unload.
 * @param {string} chatSessionId - The current session ID.
 */
export const closeSession = (chatSessionId) => {
  const data = JSON.stringify({ chatSessionId });
  navigator.sendBeacon(`${API_BASE_URL}/session/close`, new Blob([data], { type: 'application/json' }));
};