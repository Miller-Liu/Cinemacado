import React, { useEffect, useState } from 'react';
import { Helmet, HelmetProvider } from "react-helmet-async";
import { Link, useNavigate } from 'react-router-dom';
import { getAuth, onAuthStateChanged, signOut } from "firebase/auth";
import { db } from '../../firebase';
import { doc, updateDoc, arrayUnion } from "firebase/firestore";
import { useLocation } from 'react-router-dom';

export default function AddReview() {
  const [user, setUser] = useState(null);
  const [ movieName, setMovieName ] = useState('');
  const [ movieYear, setMovieYear ] = useState('');
  const [ movieRating, setMovieRating ] = useState('');
  const navigate = useNavigate();
  const location = useLocation();
  const movie = location.state?.movie;

  useEffect(() => {
    const auth = getAuth();
    const unsubscribe = onAuthStateChanged(auth, async (currentUser) => {
      setUser(currentUser);
    });
    if (movie) {
      setMovieName(movie.title);
      const fulldate = (movie.year);
      setMovieYear(fulldate);
    }
    return () => unsubscribe();
  }, [movie]);

  const onClickAddReview = async() => {
    if (!user) return;

    try {
      const userId = user.uid;
      const newReview = {
          name: movieName,
          year: movieYear,
          rating: movieRating,
      };
      await updateDoc(doc(db, "users", userId), {
          reviews: arrayUnion(newReview)
      });

      navigate(`/profile`);
  } catch (error) {
      console.error("Error adding review: ", error);
  }
};

  return (
    <HelmetProvider>
      <Helmet>
        <title>Cinemacado</title>
      </Helmet>
      <div className="landing-page">
        <h1>Add New Review</h1>
        <input type="text" placeholder="Movie" spellCheck="false" value={movieName} onChange={(e) => setMovieName(e.target.value)}/>
        <input type="text" placeholder="Year" spellCheck="false" value={movieYear} onChange={(e) => setMovieYear(e.target.value)}/>
        <input type="text" placeholder="Rating" spellCheck="false" value={movieRating} onChange={(e) => setMovieRating(e.target.value)}/>
        <button onClick={onClickAddReview}>Submit Review</button>
      </div>
    </HelmetProvider>
  );
}