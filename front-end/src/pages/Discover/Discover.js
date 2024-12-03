import './Discover.css'

import { Helmet, HelmetProvider } from "react-helmet-async";
import { useEffect, useState } from "react";
import { getAuth, onAuthStateChanged } from "firebase/auth";
import { useNavigate, useParams } from "react-router-dom";
import Movie from "../../components/Movie/Movie";

import { getGenre, getKeyword, getPerson } from './helper';
import secret from "../../secret";

export default function Discover() {
    // Allows us to redirect to another path
    const navigate = useNavigate()
    const [user, setUser] = useState(null);

    const handleAddReview = (movie) => {
        if (user) {
          navigate('/add-review', { state: { movie } });
        } else {
          // Optionally, you can redirect to login page or show a message
          alert("Please log in to add a review");
        }
      };

    const [ sortBy, setSortBy ] = useState('popularity.desc');
    const [ withCast, setWithCast ] = useState('');
    const [ withGenres, setWithGenres ] = useState('');
    const [ withKeywords, setWithKeywords ] = useState('');
    const [ year, setYear ] = useState('');

    const { discoverParams } = useParams();
    const [ discoverResult, setDiscoverResult ] = useState([]);
    const discoverMovies = async() => {
        if (discoverParams) {
            try {
                const url = `https://api.themoviedb.org/3/discover/movie?include_adult=false&include_video=false&language=en-US&page=1&${discoverParams}`;
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
                            processedResult.push({"title": item.name || item.original_title, "cover": `https://image.tmdb.org/t/p/original${item.poster_path}`, "year": item.release_date.substring(0, 4)})
                        }
                    }
                }
                
                setDiscoverResult([...processedResult]);
            } catch (error) {
                console.error('Error fetching data:', error);
            }
        }
    }

    useEffect(() => {
        const auth = getAuth();
        const unsubscribe = onAuthStateChanged(auth, (currentUser) => {
          setUser(currentUser);
        });
        
        discoverMovies();
        
        return () => unsubscribe();
      }, [discoverParams]);

    const onClickDiscover = async() => {
        const with_cast = await getPerson(withCast);
        const with_genres = await getGenre(withGenres);
        const with_keywords = await getKeyword(withKeywords);
        navigate(`/discover/sort_by=${sortBy}${with_cast}${with_genres}${with_keywords}`);
    };

    return (
        <HelmetProvider>
            <Helmet>
                <title>Discover</title>
            </Helmet>
            <div>
                <div id='DiscoverBlock'>
                    <select value={sortBy} onChange={(e) => setSortBy(e.target.value)}>
                        <option value="popularity.desc">Sort By</option>
                        <option value="popularity.desc">Most Popular</option>
                        <option value="revenue.desc">Highest Revenue</option>
                        <option value="original_title.desc">Title, A to Z</option>
                        <option value="original_title.asc">Title, Z to A</option>
                    </select>
                    <input type="text" placeholder="Cast" spellCheck="false" value={withCast} onChange={(e) => setWithCast(e.target.value)}/>
                    <input type="text" placeholder="Genres" spellCheck="false" value={withGenres} onChange={(e) => setWithGenres(e.target.value)}/>
                    <input type="text" placeholder="Keywords" spellCheck="false" value={withKeywords} onChange={(e) => setWithKeywords(e.target.value)}/>
                    <input type="number" placeholder="Year" value={year} onChange={(e) => setYear(e.target.value)}/>
                    <button onClick={onClickDiscover}>Discover</button>
                </div>
            </div>
            <br />
            <br />
            <div id="ResultsGallery" className="moviesContainer">
                {discoverResult.map((item, index) => (
                    <Movie
                        key={index}
                        title={item.title}
                        cover={item.cover}
                        year={item.year}
                        user={user}
                        onAddReview={() => handleAddReview(item)}
                    />
                ))}
            </div>
        </HelmetProvider>
    );
}