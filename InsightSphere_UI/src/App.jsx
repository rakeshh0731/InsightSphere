import React, { useState, useEffect } from 'react';
import LandingPage from './components/LandingPage';
import Chatbot from './components/Chatbot';
import { startSession, closeSession } from './api/chatService';
import './App.css';

function App() {
  const [user, setUser] = useState(null);
  const [sessionId, setSessionId] = useState(null);
  const [error, setError] = useState(null);

  const handleStart = async (name) => {
    try {
      setError(null);
      const id = await startSession();
      setSessionId(id);
      setUser({ name });
    } catch (err) {
      console.error("Failed to start session:", err);
      setError("Could not connect to the chatbot service. Please try again later.");
    }
  };

  useEffect(() => {
    const handleTabClose = (event) => {
      if (sessionId) {
        closeSession(sessionId);
      }
    };

    window.addEventListener('beforeunload', handleTabClose);

    return () => {
      window.removeEventListener('beforeunload', handleTabClose);
    };
  }, [sessionId]);

  return (
    <div className="app-container">
      {!user ? (
        <LandingPage onStart={handleStart} error={error} />
      ) : (
        <div className="main-content">
          <h1>Welcome to InsightSphere, {user.name}!</h1>
          <p>The InsightSphere assistant is ready to help in the bottom right corner.</p>
          <Chatbot sessionId={sessionId} />
        </div>
      )}
    </div>
  );
}

export default App;