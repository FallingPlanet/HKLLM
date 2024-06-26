import React, { useState, useEffect, useCallback } from 'react';
import useWebSocket, { ReadyState } from 'react-use-websocket';

function App() {
  const [messageHistory, setMessageHistory] = useState([]);
  const [inputMessage, setInputMessage] = useState('');
  const [clientId] = useState(`user-${Math.floor(Math.random() * 1000)}`);

  const { sendMessage, lastMessage, readyState } = useWebSocket(`ws://localhost:8000/ws/${clientId}`);

  useEffect(() => {
    if (lastMessage !== null) {
      setMessageHistory((prev) => prev.concat(lastMessage));
    }
  }, [lastMessage, setMessageHistory]);

  const handleClickSendMessage = useCallback(() => {
    sendMessage(JSON.stringify({
      type: 'chat_request',
      prompt: inputMessage
    }));
    setInputMessage('');
  }, [sendMessage, inputMessage]);

  const connectionStatus = {
    [ReadyState.CONNECTING]: 'Connecting',
    [ReadyState.OPEN]: 'Open',
    [ReadyState.CLOSING]: 'Closing',
    [ReadyState.CLOSED]: 'Closed',
    [ReadyState.UNINSTANTIATED]: 'Uninstantiated',
  }[readyState];

  return (
    <div>
      <span>The WebSocket is currently {connectionStatus}</span>
      <input
        type="text"
        value={inputMessage}
        onChange={(e) => setInputMessage(e.target.value)}
      />
      <button onClick={handleClickSendMessage} disabled={readyState !== ReadyState.OPEN}>
        Send
      </button>
      <ul>
        {messageHistory.map((message, idx) => (
          <li key={idx}>{message ? message.data : null}</li>
        ))}
      </ul>
    </div>
  );
}

export default App;