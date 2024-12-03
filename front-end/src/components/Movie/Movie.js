import React, { useState } from 'react';
import "./Movie.css";

export default function Movie({ title, cover, user, year, onAddReview }) {
  const [isHovered, setIsHovered] = useState(false);

  return (
    <div 
      className="container"
      onMouseEnter={() => setIsHovered(true)}
      onMouseLeave={() => setIsHovered(false)}
    >
      <img src={cover} alt={title} className="poster" />
      <div className="overlay">
        <h3 className="title">{title}</h3>
        {isHovered && user && (
            <button 
            className="add-review-button"
            onClick={() => onAddReview()}
            >
            Add Review
            </button>
        )}
         </div>
    </div>
  );
}