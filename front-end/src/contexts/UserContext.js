import React, { createContext, useState, useEffect } from 'react';
import { auth } from '../firebase';
import { onAuthStateChanged } from "firebase/auth";

export const UserContext = createContext();

export const UserProvider = ({ children }) => {
  const [user, setUser] = useState(null);
  const [loading, setLoading] = useState(true);

  useEffect(() => {
    const unsubscribe = onAuthStateChanged(auth, (user) => {
      setUser(user);
      setLoading(false);
    });

    return () => unsubscribe();
  }, []);

  return (
    <UserContext.Provider value={{ user, loading }}>
      {children}
    </UserContext.Provider>
  );
};

// src/contexts/User.js
// import React, { createContext, useState, useEffect } from 'react';
// import { auth } from '../firebase';
// import { onAuthStateChanged } from 'firebase/auth';

// export const UserContext = createContext();

// export const UserProvider = ({ children }) => {
//   const [user, setUser] = useState(null);

//   useEffect(() => {
//     const unsubscribe = onAuthStateChanged(auth, (currentUser) => {
//       setUser(currentUser);
//     });

//     return () => unsubscribe();
//   }, []);

//   return (
//     <UserContext.Provider value={{ user }}>
//       {children}
//     </UserContext.Provider>
//   );
// };