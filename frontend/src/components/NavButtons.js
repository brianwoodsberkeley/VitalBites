import React from 'react';

function NavButtons() {
  return (
    <>
      <a
        href="https://github.com/brianwoodsberkeley/VitalBites/issues?q=is%3Aissue%20state%3Aopen%20label%3Abug"
        target="_blank"
        rel="noopener noreferrer"
        className="bug-btn"
        title="Report a bug"
      >
        <svg width="16" height="16" viewBox="0 0 16 16" fill="currentColor">
          <path d="M4.355.522a.5.5 0 01.623.333l.291.956A5 5 0 018 1c1.007 0 1.946.298 2.731.811l.29-.956a.5.5 0 11.957.29l-.41 1.352A5 5 0 0113 4h.5a.5.5 0 010 1H13a5 5 0 01-.034.5H14.5a.5.5 0 010 1h-1.757a5.5 5.5 0 01-9.486 0H1.5a.5.5 0 010-1h1.534A5 5 0 013 5h-.5a.5.5 0 010-1H3a5 5 0 011.432-2.455l-.41-1.352a.5.5 0 01.333-.623zM8 2a4 4 0 00-4 4 4.5 4.5 0 008.945.5H12a4 4 0 00-4-4.5V2zM4.5 5.5a.5.5 0 000 1h7a.5.5 0 000-1h-7z"/>
        </svg>
        Report Bug
      </a>
      <a
        href="https://github.com/brianwoodsberkeley/VitalBites/issues?q=is%3Aissue%20state%3Aopen%20label%3Aenhancement"
        target="_blank"
        rel="noopener noreferrer"
        className="feature-btn"
        title="Request a feature"
      >
        <svg width="16" height="16" viewBox="0 0 16 16" fill="currentColor">
          <path d="M8 0a.5.5 0 01.5.5v2a.5.5 0 01-1 0v-2A.5.5 0 018 0zM3.146 2.646a.5.5 0 01.708 0l1.414 1.414a.5.5 0 11-.708.708L3.146 3.354a.5.5 0 010-.708zM11.44 4.06a.5.5 0 010-.708l1.414-1.414a.5.5 0 01.708.708L12.148 4.06a.5.5 0 01-.708 0zM8 5a3 3 0 100 6 3 3 0 000-6zM0 8a.5.5 0 01.5-.5h2a.5.5 0 010 1h-2A.5.5 0 010 8zm13 0a.5.5 0 01.5-.5h2a.5.5 0 010 1h-2A.5.5 0 0113 8zM4.768 11.94a.5.5 0 010 .708l-1.414 1.414a.5.5 0 11-.708-.708l1.414-1.414a.5.5 0 01.708 0zm6.464 0a.5.5 0 01.708 0l1.414 1.414a.5.5 0 01-.708.708l-1.414-1.414a.5.5 0 010-.708zM8 13a.5.5 0 01.5.5v2a.5.5 0 01-1 0v-2A.5.5 0 018 13z"/>
        </svg>
        Request Feature
      </a>
    </>
  );
}

export default NavButtons;
