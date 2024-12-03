import React from 'react';
import './Review.css';

const Review = ({ review }) => {
  return (
    <div className="review-card">
      <h3 className="movie-title">{review.name}</h3>
      <div className="movie-details">
        <span className="movie-year">{review.year}</span>
        <span className="movie-rating">Rating: {review.rating}/5</span>
      </div>
      {review.comment && (
        <p className="movie-comment">{review.comment}</p>
      )}
    </div>
  );
};

export default Review;