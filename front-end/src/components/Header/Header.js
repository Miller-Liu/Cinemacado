// import Link from "next/link";
import { Link } from 'react-router-dom';
import "./Header.css";
import React, { useState } from "react";

function Header() {
    const [ expanded, setExpanded ] = useState(false);
    
    const toggleMenu = () => {
        setExpanded(!expanded);
    }

    // Function to handle click on menu item and close the expanded menu
    const closeAfterClick = () => {
        setExpanded(false); // Set expanded to false to close the menu
    };

    return (
        <div className="header">
            {expanded && (
                <div className="expanded-menu">
                    <Link to="/" className="menu-button" onClick={closeAfterClick}>
                        <i className="fas fa-home"></i>
                        </Link>
                    <Link to="/profile" className="menu-button" onClick={closeAfterClick}>
                        <i className="fas fa-user"></i>
                    </Link>
                    <Link to="/search" className="menu-button" onClick={closeAfterClick}>
                        <i className="fas fa-search"></i>
                    </Link>
                    <Link to="/discover" className="menu-button" onClick={closeAfterClick}>
                        <i className="fas fa-camera-retro"></i>
                    </Link>
                </div>
            )}
            <button className="menu-button" onClick={toggleMenu}>
                <i  className="fas fa-bars"></i>
            </button>
        </div>
    );
}

export default Header;