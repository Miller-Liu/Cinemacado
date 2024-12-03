import './Search.css'

import { Helmet, HelmetProvider } from "react-helmet-async";
import { useEffect, useState } from "react";
import { useNavigate, useParams } from "react-router-dom";
import Movie from "../../components/Movie/Movie";

import secret from "../../secret";

export default function Search() {
    // Allows us to redirect to another path
    const navigate = useNavigate()

    // Search up movie by calling TMDB API using search parameters
    const { searchParams } = useParams();
    const [searchResult, setSearchResult] = useState([]);
    const searchMovies = async() => {
        if (searchParams) {
            try {
                const url = `https://api.themoviedb.org/3/search/multi?query=${searchParams}&include_adult=${false}&language=en-US&page=${1}'`;
                const options = {
                    method: 'GET',
                    headers: {
                      accept: 'application/json',
                      Authorization: `Bearer ${secret.TMDB.API_read_access_token}`
                    }
                };
    
                const response = await fetch(url, options);
                
                if (!response.ok) {
                  throw new Error(`HTTP error! Status: ${response.status}`);
                }
        
                const result = await response.json();
                const processedResult = [];
                
                if (result.total_results > 0) {
                    for (const item of result.results) {
                        if ((item.name || item.original_title) && item.poster_path) {
                            processedResult.push({"title": item.name || item.original_title, "cover": `https://image.tmdb.org/t/p/original${item.poster_path}`})
                        }
                    }
                }
                
                setSearchResult([...processedResult]);
            } catch (error) {
                console.error('Error fetching data:', error);
            }
        }
    };
    useEffect(() => {searchMovies()}, [searchParams]);

    useEffect(() => {console.log(searchResult)}, [searchResult]);

    const handleKeyPress = (event) => {
        if (event.key === 'Enter') {
            navigate(`/search/${searchTitle}`);
        }
    };

    // Tracks what's entered in the search bar
    const [searchTitle, setSearchTitle] = useState(searchParams || "");

    return (
        <HelmetProvider>
            <Helmet>
                <title>Searching for...</title>
            </Helmet>
            <div>
                <div id='SearchBlock'>
                    <div id='logo-container'>
                        <svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 512 512"><path d="M416 208c0 45.9-14.9 88.3-40 122.7L502.6 457.4c12.5 12.5 12.5 32.8 0 45.3s-32.8 12.5-45.3 0L330.7 376c-34.4 25.2-76.8 40-122.7 40C93.1 416 0 322.9 0 208S93.1 0 208 0S416 93.1 416 208zM208 352a144 144 0 1 0 0-288 144 144 0 1 0 0 288z"/></svg>
                    </div>
                    <div id='input-container'>
                        <input type="text" placeholder="Please Enter Title" spellCheck="false" value={searchTitle} onChange={(e) => setSearchTitle(e.target.value)} onKeyDown={handleKeyPress}/>
                        <div id='submitButton' onClick={() => navigate(`/search/${searchTitle}`)}>
                            <svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 448 512"><path d="M438.6 278.6c12.5-12.5 12.5-32.8 0-45.3l-160-160c-12.5-12.5-32.8-12.5-45.3 0s-12.5 32.8 0 45.3L338.8 224 32 224c-17.7 0-32 14.3-32 32s14.3 32 32 32l306.7 0L233.4 393.4c-12.5 12.5-12.5 32.8 0 45.3s32.8 12.5 45.3 0l160-160z"/></svg>
                        </div>
                    </div>
                </div>
            </div>
            <br/>
            <br/>
            <div id="ResultsGallery" className="moviesContainer">
                {searchResult.map((item, index) => (
                    <Movie
                        key={index}
                        title={item.title}
                        cover={item.cover}
                    />
                ))}
            </div>
        </HelmetProvider>
    );
}