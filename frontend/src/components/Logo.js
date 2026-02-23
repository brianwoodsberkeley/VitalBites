import React from 'react';

function Logo({ height = 40, className = '' }) {
  return (
    <img
      src="/vital_bites_logo.jpg"
      alt="VitalFoods AI"
      className={className}
      style={{
        height,
        width: 'auto',
        objectFit: 'contain',
        borderRadius: 6,
      }}
    />
  );
}

export default Logo;
