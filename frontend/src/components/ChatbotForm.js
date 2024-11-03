import React, { useState } from 'react';
import ResponseDisplay from './ResponseDisplay';

const apiBaseUrl = "http://127.0.0.1:8000/api/v1";

function ChatbotForm() {
  const [indexName, setIndexName] = useState('');
  const [filePaths, setFilePaths] = useState('');
  const [question, setQuestion] = useState('');
  const [responseMessage, setResponseMessage] = useState('');

  const showResponse = (message) => {
    setResponseMessage(typeof message === "string" ? message : JSON.stringify(message, null, 2));
  };

  const handleInitialize = async () => {
    if (!indexName) {
      alert("Please provide an index name.");
      return;
    }

    try {
      const response = await fetch(`${apiBaseUrl}/initialize`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ index_name: indexName })
      });

      const data = await response.json();
      showResponse(response.ok ? data.message : `Error: ${data.detail || JSON.stringify(data)}`);
    } catch (error) {
      showResponse(`Error: ${error.message}`);
    }
  };

  const handleTrain = async () => {
    const filePathArray = filePaths.split(',').map(path => path.trim());

    if (!indexName || filePathArray.length === 0 || filePathArray[0] === "") {
      alert("Please provide an index name and file paths.");
      return;
    }

    try {
      const response = await fetch(`${apiBaseUrl}/train`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ index_name: indexName, file_paths: filePathArray })
      });

      const data = await response.json();
      showResponse(response.ok ? data.message : `Error: ${data.detail}`);
    } catch (error) {
      showResponse(`Error: ${error.message}`);
    }
  };

  const handleAsk = async () => {
    if (!indexName || !question) {
      alert("Please provide both an index name and a question.");
      return;
    }

    try {
      const response = await fetch(`${apiBaseUrl}/ask`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ index_name: indexName, question: question })
      });

      const data = await response.json();
      const answer = typeof data.answer === "object" ? JSON.stringify(data.answer, null, 2) : data.answer;
      showResponse(response.ok ? `Answer: ${answer}` : `Error: ${data.detail}`);
    } catch (error) {
      showResponse(`Error: ${error.message}`);
    }
  };

  return (
    <div>
      <label>
        Index Name:
        <input
          type="text"
          value={indexName}
          onChange={(e) => setIndexName(e.target.value)}
          placeholder="Enter index name"
          required
        />
      </label>

      <label>
        File Paths (comma-separated for multiple files):
        <input
          type="text"
          value={filePaths}
          onChange={(e) => setFilePaths(e.target.value)}
          placeholder="Enter file paths"
          required
        />
      </label>

      <label>
        Question:
        <textarea
          value={question}
          onChange={(e) => setQuestion(e.target.value)}
          rows="4"
          placeholder="Type your question here"
          required
        />
      </label>

      <button onClick={handleInitialize}>Initialize ChatBot</button>
      <button onClick={handleTrain}>Train ChatBot</button>
      <button onClick={handleAsk}>Ask ChatBot</button>

      <ResponseDisplay message={responseMessage} />
    </div>
  );
}

export default ChatbotForm;
