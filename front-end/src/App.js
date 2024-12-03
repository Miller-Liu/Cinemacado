import React from 'react';
import Header from './components/Header/Header';
import Search from './pages/Search/Search';
import Discover from './pages/Discover/Discover';
import Landing from './pages/Landing/Landing';
import Login from './pages/Login/Login';
import Signup from './pages/Signup/Signup';
import { UserProvider } from './contexts/UserContext';
import BackendTest from './pages/BackendTest/BackendTest';
import Profile from './pages/Profile/Profile';
import './App.css';
import { BrowserRouter as Router, Routes, Route } from 'react-router-dom';
import AddReview from './pages/AddReview/AddReview';

export default function App() {
  return (
    <UserProvider>
    {/* <Router> */}
    <div>
      <Header />
      <Routes>
        <Route path='/search'>
          <Route path=':searchParams?' element={<Search/>}/>
        </Route>
        <Route path='/discover'>
          <Route path=':discoverParams?' element={<Discover/>}/>
        </Route>
        <Route path='/' element={<Landing/>}/>
        <Route path='/profile' element={<Profile/>}/>
        <Route path='/login' element={<Login/>}/>
        <Route path='/signup' element={<Signup/>}/>
        <Route path='/backend-test' element={<BackendTest/>}/>
        <Route path='/add-review' element={<AddReview/>}/>
      </Routes>
    </div>
    {/* </Router> */}
    </UserProvider>
  );
}