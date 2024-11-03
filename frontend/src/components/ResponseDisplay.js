import React from 'react';

function ResponseDisplay({ message }) {
  return (
    <div style={{ marginTop: "20px", padding: "10px", backgroundColor: "#d1e7dd", borderRadius: "5px", border: "1px solid #badbcc" }}>
      {message}
    </div>
  );
}

export default ResponseDisplay;
