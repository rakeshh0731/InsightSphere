import React, { useState } from 'react';
import './LandingPage.css';

function LandingPage({ onStart, error }) {
  const [name, setName] = useState('');
  const [isLoading, setIsLoading] = useState(false);

  const handleSubmit = async (e) => {
    e.preventDefault();
    if (name.trim() && !isLoading) {
      setIsLoading(true);
      await onStart(name.trim());
      // No need to setIsLoading(false) as the component will unmount
    }
  };

  return (
    <div className="landing-container">
      <div className="landing-box">
        <h1>Welcome to InsightSphere</h1>
        <p>Your AI-powered analytics engine.</p>
        <form onSubmit={handleSubmit}>
          <input
            type="text"
            value={name}
            onChange={(e) => setName(e.target.value)}
            placeholder="Please enter your name"
            aria-label="Your Name"
            required
          />
          <button type="submit" disabled={isLoading}>
            {isLoading ? 'Connecting...' : 'Start Chatting'}
          </button>
        </form>
        {error && <p className="error-message">{error}</p>}
      </div>
    </div>
  );
}

export default LandingPage;