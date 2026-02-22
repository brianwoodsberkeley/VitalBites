import React, { useState } from 'react';

const THROBBER_IMAGES = [
  '/throbbers/giorgio-trovato-YQVpD52cKpo-unsplash.jpg',
  '/throbbers/amjd-rdwan-8du-1nR9OkM-unsplash.jpg',
  '/throbbers/trac-vu-EovrNaCXeS0-unsplash.jpg',
  '/throbbers/aby-zachariah-2DfjRspDYJE-unsplash.jpg',
  '/throbbers/alien-beker-Jw3XXq4u7NU-unsplash.jpg',
  '/throbbers/nick-fewings-UHpggTYXLYE-unsplash.jpg',
  '/throbbers/halim-rox-AETahyCi2Dc-unsplash.jpg',
  '/throbbers/mustafa-akin-y5hcOQOqPJA-unsplash.jpg',
  '/throbbers/lighten-up-E0FAAaKTjkU-unsplash.jpg',
  '/throbbers/nikhil-SbX6tSMMsIk-unsplash.jpg',
  '/throbbers/sharan-pagadala-264Yk95Osm0-unsplash.jpg',
  '/throbbers/abdullah-nazeer-E3tj-mse3W4-unsplash.jpg',
];

function pickRandom() {
  return THROBBER_IMAGES[Math.floor(Math.random() * THROBBER_IMAGES.length)];
}

function Throbber({ size = 64, seed }) {
  const [image, setImage] = useState(pickRandom);

  React.useEffect(() => {
    if (seed !== undefined) {
      setImage(pickRandom());
    }
  }, [seed]);

  return (
    <div
      className="spinner"
      style={{
        width: size,
        height: size,
        backgroundImage: `url('${image}')`,
      }}
    />
  );
}

export default Throbber;
