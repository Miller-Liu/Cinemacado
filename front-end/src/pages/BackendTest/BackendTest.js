import React, { useEffect, useState } from 'react';
import { Helmet, HelmetProvider } from "react-helmet-async";

export default function BackendTest() {
  const [movies, setMovies] = useState([]);

  useEffect(() => {
    fetch('/api/movies')
      .then(response => response.json())
      .then(data => setMovies(data.movies))
      .catch(error => console.error('Error fetching movies:', error));
  }, []);

  return (
    <HelmetProvider>
      <Helmet>
        <title>Cinemacado</title>
      </Helmet>
      <div className="landing-page">
        <h1>Sample Movies</h1>
        <div className="movies-container">
          {movies.map((movie, index) => (
            <div key={index} className="movie">
              <img src={movie.cover} alt={movie.title} />
              <h3>{movie.title}</h3>
            </div>
          ))}
        </div>
      </div>
    </HelmetProvider>
  );
}