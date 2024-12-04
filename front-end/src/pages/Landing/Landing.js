import React, { useEffect, useState } from 'react';
import './Landing.css';
import { Helmet, HelmetProvider } from "react-helmet-async";
import { getAuth, onAuthStateChanged } from "firebase/auth";
import { Link } from 'react-router-dom';
// import { UserContext } from '../../contexts/UserContext'

export default function Landing() {
  const [user, setUser] = useState(null);
  const [loading, setLoading] = useState(true);
  const [results, setResults ] = useState([]);
  // const { user } = useContext(UserContext);

  useEffect(() => {
    const auth = getAuth();
    const unsubscribe = onAuthStateChanged(auth, (currentUser) => {
      setUser(currentUser);
      setLoading(false);
    });

    return () => unsubscribe();
  }, []);

  useEffect(() => {
    if (user) {
      fetch(`/user/${user.uid}`)
        .then(response => response.json())
        .then(data => setResults(data))
    }
  }, [user])

  if (loading) {
    return <div>Loading...</div>;
  }

  return (
    <HelmetProvider>
      <Helmet>
        <title>Cinemacado</title>
      </Helmet>
      <div className="landing-page">
        {user ? (
          <div className="welcome-container">
            <h1>Welcome to Cinemacado, {user.email}!</h1>
            <p>Enjoy exploring movies and sharing your reviews.</p>
          </div>
        ) : (
          <div className="auth-options">
            <h1>Welcome to Cinemacado</h1>
            <p>Please log in or sign up to continue.</p>
            <div className="auth-buttons">
              <Link to="/login" className="auth-button">Log In</Link>
              <Link to="/signup" className="auth-button">Sign Up</Link>
            </div>
          </div>
        )}
        <div>
            {results.map(element => (
                <div>{element}</div>
            ))}
        </div>
      </div>
  
    </HelmetProvider>
  );
}