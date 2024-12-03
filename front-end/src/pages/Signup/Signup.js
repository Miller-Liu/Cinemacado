import React, { useState } from 'react';
import { createUserWithEmailAndPassword } from "firebase/auth";
import { auth, db, storage } from '../../firebase';
import { uploadProfilePicture } from '../../utils/firebaseUtils';
import { ref, uploadBytes, getDownloadURL } from "firebase/storage";
import { doc, setDoc } from "firebase/firestore";
import { useNavigate, Link } from 'react-router-dom';

export default function Signup() {
  const [email, setEmail] = useState('');
  const [password, setPassword] = useState('');
  const [genres, setGenres] = useState([]);
  const [ratings, setRatings] = useState([]);
  const [profilePicture, setProfilePicture] = useState(null);
  const [error, setError] = useState('');
  const [success, setSuccess] = useState(false);
  const navigate = useNavigate();

  const handleSubmit = async (e) => {
    e.preventDefault();
    setError('');
    setSuccess(false);

    try {
      const userCredential = await createUserWithEmailAndPassword(auth, email, password);
      const user = userCredential.user;

      let profilePictureUrl = null;
      if (profilePicture) {
        const storageRef = ref(storage, `profilePictures/${user.uid}`);
        await uploadBytes(storageRef, profilePicture);
        profilePictureUrl = await getDownloadURL(storageRef);
      }
      
      // Save user data to Firestore
      await setDoc(doc(db, "users", user.uid), {
        email: email,
        genres: genres,
        ratings: ratings,
        profilePicture: profilePictureUrl
      });

      setSuccess(true);
      // Clear form fields
      setEmail('');
      setPassword('');
      setGenres([]);
      setRatings([])
      setProfilePicture(null);
      navigate('/profile');
    } catch (error) {
      console.error("Error during signup:", error);
      setError(error.message);
    }
  };

  const handleGenreChange = (e) => {
    const genre = e.target.value;
    setGenres(prev => 
      prev.includes(genre) ? prev.filter(g => g !== genre) : [...prev, genre]
    );
  };

  const handleFileChange = (e) => {
    setProfilePicture(e.target.files[0]);
  };

  return (
    <div className="signup-container">
      <h2>Sign Up</h2>
      {success && (
        <div className="success-message">
          <p>Sign up successful! Your account has been created.</p>
          <Link to="/login" className="login-link">Go to Login</Link>
        </div>
      )}
      {error && <p className="error-message">{error}</p>}
      <form onSubmit={handleSubmit}>
        <input
          type="email"
          placeholder="Email"
          value={email}
          onChange={(e) => setEmail(e.target.value)}
          required
        />
        <input
          type="password"
          placeholder="Password"
          value={password}
          onChange={(e) => setPassword(e.target.value)}
          required
        />
        <div>
          <p>Select your favorite genres:</p>
          {['romance', 'comedy', 'drama', 'action'].map(genre => (
            <label key={genre}>
              <input
                type="checkbox"
                value={genre}
                checked={genres.includes(genre)}
                onChange={handleGenreChange}
              />
              {genre}
            </label>
          ))}
        </div>
        {/* <input
          type="file"
          accept="image/*"
          onChange={handleFileChange}
        /> */}
        <button type="submit">Sign Up</button>
      </form>
      <p>Already have an account? <Link to="/login">Login</Link></p>
    </div>
  );
}