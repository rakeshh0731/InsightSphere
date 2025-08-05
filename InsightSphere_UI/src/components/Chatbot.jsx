import React, { useState, useEffect, useRef } from 'react';
import ReactMarkdown from 'react-markdown';
import remarkGfm from 'remark-gfm';
import { sendMessage, resetSession } from '../api/chatService';
import './Chatbot.css';

// Main icon for the chatbot toggle button
const ChatbotIcon = () => (
  <svg xmlns="http://www.w3.org/2000/svg" width="24" height="24" viewBox="0 0 24 24" fill="none" stroke="white" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round">
    <path d="M21 15a2 2 0 0 1-2 2H7l-4 4V5a2 2 0 0 1 2-2h14a2 2 0 0 1 2 2z"></path>
  </svg>
);

// Icon for closing the chat window
const CloseIcon = () => (
    <svg xmlns="http://www.w3.org/2000/svg" width="24" height="24" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round">
        <line x1="18" y1="6" x2="6" y2="18"></line><line x1="6" y1="6" x2="18" y2="18"></line>
    </svg>
);

// NEW: A more appropriate "trash" icon for clearing the chat
const TrashIcon = () => (
    <svg xmlns="http://www.w3.org/2000/svg" width="18" height="18" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round">
        <polyline points="3 6 5 6 21 6"></polyline>
        <path d="M19 6v14a2 2 0 0 1-2 2H7a2 2 0 0 1-2-2V6m3 0V4a2 2 0 0 1 2-2h4a2 2 0 0 1 2 2v2"></path>
        <line x1="10" y1="11" x2="10" y2="17"></line>
        <line x1="14" y1="11" x2="14" y2="17"></line>
    </svg>
);

// NEW: Avatar for the Bot
const BotAvatarIcon = () => (
    <div className="avatar bot-avatar">
        <svg xmlns="http://www.w3.org/2000/svg" width="20" height="20" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round">
            <rect x="4" y="4" width="16" height="16" rx="2" ry="2"></rect>
            <rect x="9" y="9" width="6" height="6"></rect>
            <line x1="9" y1="1" x2="9" y2="4"></line><line x1="15" y1="1" x2="15" y2="4"></line>
            <line x1="9" y1="20" x2="9" y2="23"></line><line x1="15" y1="20" x2="15" y2="23"></line>
            <line x1="20" y1="9" x2="23" y2="9"></line><line x1="20" y1="14" x2="23" y2="14"></line>
            <line x1="1" y1="9" x2="4" y2="9"></line><line x1="1" y1="14" x2="4" y2="14"></line>
        </svg>
    </div>
);

// NEW: Avatar for the User
const UserAvatarIcon = () => (
    <div className="avatar user-avatar">
        <svg xmlns="http://www.w3.org/2000/svg" width="20" height="20" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round">
            <path d="M20 21v-2a4 4 0 0 0-4-4H8a4 4 0 0 0-4 4v2"></path>
            <circle cx="12" cy="7" r="4"></circle>
        </svg>
    </div>
);

function Chatbot({ sessionId }) {
  const [isOpen, setIsOpen] = useState(false);
  const [messages, setMessages] = useState([
    { sender: 'bot', text: 'Hello! How can I help you with your documents today?' },
  ]);
  const [input, setInput] = useState('');
  const [isLoading, setIsLoading] = useState(false);
  const messagesEndRef = useRef(null);

  const scrollToBottom = () => {
    messagesEndRef.current?.scrollIntoView({ behavior: "smooth" });
  };

  useEffect(scrollToBottom, [messages, isLoading]);

  const handleSendMessage = async (e) => {
    e.preventDefault();
    if (input.trim() && !isLoading) {
      const userMessage = { sender: 'user', text: input.trim() };
      setMessages(prev => [...prev, userMessage]);
      setInput('');
      setIsLoading(true);

      try {
        const reply = await sendMessage(sessionId, userMessage.text);
        setMessages(prev => [...prev, { sender: 'bot', text: reply }]);
      } catch (error) {
        console.error("Failed to send message:", error);
        setMessages(prev => [...prev, { sender: 'bot', text: "Sorry, I'm having trouble connecting. Please try logging again." }]);
      } finally {
        setIsLoading(false);
      }
    }
  };

  const handleReset = async () => {
    if (window.confirm("Are you sure you want to clear the conversation? This action cannot be undone.")) {
        try {
            await resetSession(sessionId);
            setMessages([{ sender: 'bot', text: 'Conversation cleared. How can I help you?' }]);
        } catch (error) {
            console.error("Failed to reset session:", error);
            setMessages(prev => [...prev, { sender: 'bot', text: "Sorry, I couldn't clear the session." }]);
        }
    }
  };

  return (
    <>
      <div className={`chatbot-window ${isOpen ? 'open' : ''}`}>
        <div className="chatbot-header">
          <h2>InsightSphere Assistant</h2>
          <div className="header-buttons">
            <button onClick={handleReset} className="icon-button" aria-label="Clear Conversation"><TrashIcon /></button>
            <button onClick={() => setIsOpen(false)} className="icon-button" aria-label="Close Chat"><CloseIcon /></button>
          </div>
        </div>
        <div className="chatbot-messages">
          {messages.map((msg, index) => (
            <div key={index} className={`message-row ${msg.sender}`}>
              {msg.sender === 'bot' && <BotAvatarIcon />}
              <div className={`message ${msg.sender}`}>
                <ReactMarkdown remarkPlugins={[remarkGfm]}>
                  {msg.text}
                </ReactMarkdown>
              </div>
              {msg.sender === 'user' && <UserAvatarIcon />}
            </div>
          ))}
          {isLoading && (
            <div className="message-row bot">
              <BotAvatarIcon />
              <div className="message bot typing-indicator">
                <span></span><span></span><span></span>
              </div>
            </div>
          )}
          <div ref={messagesEndRef} />
        </div>
        <form className="chatbot-input-form" onSubmit={handleSendMessage}>
          <input
            type="text"
            value={input}
            onChange={(e) => setInput(e.target.value)}
            placeholder="Type your message..."
            disabled={isLoading}
          />
          <button type="submit" disabled={isLoading}>Send</button>
        </form>
      </div>
      <button className={`chatbot-toggle-button ${isOpen ? 'hidden' : ''}`} onClick={() => setIsOpen(true)} aria-label="Open Chat">
        <ChatbotIcon />
      </button>
    </>
  );
}

export default Chatbot;