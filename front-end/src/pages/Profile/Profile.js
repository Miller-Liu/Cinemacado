import React, { useEffect, useState, useContext } from 'react';
import { getAuth, onAuthStateChanged, signOut } from "firebase/auth";
import { doc, getDoc, updateDoc } from "firebase/firestore";
import { getStorage, ref, uploadBytes, getDownloadURL } from "firebase/storage";
import { db, storage } from '../../firebase';
import { Link, useNavigate } from 'react-router-dom';
import './Profile.css';
import { Helmet, HelmetProvider } from "react-helmet-async";
import Review from "../../components/Review/Review";

export default function Profile() {
  const [user, setUser] = useState(null);
  // const { user } = useContext(UserContext);
  const [profileData, setProfileData] = useState(null);
  const [loading, setLoading] = useState(true);
  const [isEditing, setIsEditing] = useState(false);
  const [newProfilePicture, setNewProfilePicture] = useState(null);
  const [newGenres, setNewGenres] = useState([]);
  const [newRatings, setNewRatings] = useState([]);
  const navigate = useNavigate();


  const handleLogout = async () => {
    const auth = getAuth();
    try {
      await signOut(auth);
      window.location.href = '/';
    } catch (error) {
      console.error("Error signing out: ", error);
    }
  };

  useEffect(() => {
    const auth = getAuth();
    const unsubscribe = onAuthStateChanged(auth, async (currentUser) => {
      setUser(currentUser);
      if (currentUser) {
        try {
          const userDoc = await getDoc(doc(db, "users", currentUser.uid));
          if (userDoc.exists()) {
            const data = userDoc.data();
            setProfileData(data);
            setNewGenres(data.genres || []);
          } else {
            console.log("No such document!");
          }
        } catch (error) {
          console.error("Error fetching user data:", error);
        }
      }
      setLoading(false);
    });
  
    return () => unsubscribe();
  }, []);

  const handleEditToggle = () => {
    setIsEditing(!isEditing);
  };

  const handleAddReview = () => {
    navigate('/add-review');
  };

  const handleGenreChange = (genre) => {
    setNewGenres(prev => 
      prev.includes(genre) ? prev.filter(g => g !== genre) : [...prev, genre]
    );
  };

  const handleProfilePictureChange = (e) => {
    if (e.target.files[0]) {
      setNewProfilePicture(e.target.files[0]);
    }
  };

  const handleSaveChanges = async () => {
    if (!user) return;

    try {
      let profilePictureUrl = profileData.profilePicture;

      if (newProfilePicture) {
        const storage = getStorage();
        const storageRef = ref(storage, `profilePictures/${user.uid}`);
        await uploadBytes(storageRef, newProfilePicture);
        profilePictureUrl = await getDownloadURL(storageRef);
      }

      await updateDoc(doc(db, "users", user.uid), {
        genres: newGenres,
        profilePicture: profilePictureUrl
      });

      setProfileData(prev => ({
        ...prev,
        genres: newGenres,
        profilePicture: profilePictureUrl
      }));

      setIsEditing(false);
    } catch (error) {
      console.error("Error updating profile:", error);
    }
  };

  if (loading) {
    return <div className="loading">Loading...</div>;
  }

  if (!user) {
    return (
      <HelmetProvider>
        <Helmet>
          <title>Profile - Cinemacado</title>
        </Helmet>
        <div className="profile-container">
          <h1>Welcome to Cinemacado</h1>
          <p>Please log in or sign up to view your profile.</p>
          <div className="auth-buttons">
            <Link to="/login" className="auth-button">Log In</Link>
            <Link to="/signup" className="auth-button">Sign Up</Link>
          </div>
        </div>
      </HelmetProvider>
    );
  }

  return (
    <HelmetProvider>
      <Helmet>
        <title>{user.email}'s Profile - Cinemacado</title>
      </Helmet>
      <div className="profile-container">
        <h1>Your Profile</h1>
        {profileData && (
          <>
            <div className="profile-picture">
              <img src={profileData.profilePicture || '/avo.jpg'} alt="Profile" />
              {/* {isEditing && (
                <input type="file" onChange={handleProfilePictureChange} accept="image/*" />
              )} */}
            </div>
            <div className="profile-info">
              <h2>{user.email}</h2>
              <h3>Favorite Genres:</h3>
              {isEditing ? (
                <div className="genre-checkboxes">
                  {['romance', 'comedy', 'drama', 'action'].map(genre => (
                    <label key={genre}>
                      <input
                        type="checkbox"
                        checked={newGenres.includes(genre)}
                        onChange={() => handleGenreChange(genre)}
                      />
                      {genre}
                    </label>
                  ))}
                </div>
              ) : (
                <ul className="genre-list">
                  {profileData.genres.map((genre, index) => (
                    <li key={index} className="genre-item">{genre}</li>
                  ))}
                </ul>
              )}
              <h2>Your Reviews</h2>
    <div className="reviews-container">
      {(profileData.reviews || []).map((review, index) => (
        <Review key={index} review={review} />
      ))}
    </div>
            </div>
          </>
        )}
        {isEditing ? (
          <button onClick={handleSaveChanges} className="edit-button">Save Changes</button>
        ) : (
          <button onClick={handleEditToggle} className="edit-button">Edit Profile</button>
        )}
        <button onClick={handleAddReview} className="edit-button">Add review</button>
        <button onClick={handleLogout} className="logout-button">Log Out</button>
      </div>
    </HelmetProvider>
  );
}

// import React, { useEffect, useState, useContext } from 'react';
// import { UserContext } from '../../contexts/UserContext';
// import { doc, getDoc, updateDoc } from "firebase/firestore";
// import { ref, uploadBytes, getDownloadURL } from "firebase/storage";
// import { db, storage } from '../../firebase';
// import { Link } from 'react-router-dom';
// import './Profile.css';

// export default function Profile() {
//   const { user } = useContext(UserContext);
//   const [profileData, setProfileData] = useState(null);
//   const [isEditing, setIsEditing] = useState(false);
//   const [newGenres, setNewGenres] = useState([]);
//   const [newProfilePicture, setNewProfilePicture] = useState(null);

//   useEffect(() => {
//     if (user) {
//       const fetchProfileData = async () => {
//         const docRef = doc(db, "users", user.uid);
//         const docSnap = await getDoc(docRef);
//         if (docSnap.exists()) {
//           setProfileData(docSnap.data());
//           setNewGenres(docSnap.data().genres || []);
//         }
//       };
//       fetchProfileData();
//     }
//   }, [user]);

//   const handleSave = async () => {
//     if (!user) return;

//     let profilePictureUrl = profileData.profilePicture;

//     if (newProfilePicture) {
//       const storageRef = ref(storage, `profilePictures/${user.uid}`);
//       await uploadBytes(storageRef, newProfilePicture);
//       profilePictureUrl = await getDownloadURL(storageRef);
//     }

//     await updateDoc(doc(db, "users", user.uid), {
//       genres: newGenres,
//       profilePicture: profilePictureUrl
//     });

//     setProfileData(prev => ({ ...prev, genres: newGenres, profilePicture: profilePictureUrl }));
//     setIsEditing(false);
//   };

//   if (!user) {
//     return (
//       <div className="profile-container">
//         <h1>You are not logged in</h1>
//         <Link to="/login" className="auth-button">Log In</Link>
//         <Link to="/signup" className="auth-button">Sign Up</Link>
//       </div>
//     );
//   }

//   return (
//     <div className="profile-container">
//       <h1>Your Profile</h1>
//       {profileData && (
//         <>
//           <img src={profileData.profilePicture || '/default-avatar.png'} alt="Profile" />
//           <p>Email: {user.email}</p>
//           <h3>Favorite Genres:</h3>
//           {isEditing ? (
//             <>
//               {['romance', 'comedy', 'drama', 'action'].map(genre => (
//                 <label key={genre}>
//                   <input
//                     type="checkbox"
//                     checked={newGenres.includes(genre)}
//                     onChange={() => setNewGenres(prev => 
//                       prev.includes(genre) ? prev.filter(g => g !== genre) : [...prev, genre]
//                     )}
//                   />
//                   {genre}
//                 </label>
//               ))}
//               <input type="file" onChange={(e) => setNewProfilePicture(e.target.files[0])} />
//               <button onClick={handleSave}>Save</button>
//             </>
//           ) : (
//             <>
//               <ul>
//                 {profileData.genres.map(genre => <li key={genre}>{genre}</li>)}
//               </ul>
//               <button onClick={() => setIsEditing(true)}>Edit Profile</button>
//             </>
//           )}
//         </>
//       )}
//     </div>
//   );
// }