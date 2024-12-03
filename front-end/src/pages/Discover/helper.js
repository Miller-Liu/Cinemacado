
import secret from "../../secret";

export const getPerson = async(name) => {
    if (name) {
        try {
            const url = `https://api.themoviedb.org/3/search/person?query=${name}&include_adult=false&language=en-US&page=1`;
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
            const ids = [];

            if (result.total_results > 0) {
                for (const item of result.results) {
                    if (item.name) {
                        ids.push(item.id);
                    }
                }
            }
            return "&with_cast=" + ids.join("|");
        } catch (error) {
            console.error('Error fetching data:', error);
        }
    }
    return "";
}

export const getGenre = async(genre) => {
    if (genre) {
        try {
            const url = 'https://api.themoviedb.org/3/genre/movie/list?language=en';
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
            const ids = [];

            for (const item of result.genres) {
                if (item.name.toLowerCase() == genre.toLowerCase()) {
                    ids.push(item.id);
                }
            }

            if (ids.length) {
                return "&with_genres=" + ids.join(",");
            }

            return "";
        } catch (error) {
            console.error('Error fetching data:', error);
        }
    }
    return "";
}


export const getKeyword = async(keyword) => {
    if (keyword) {
        try {
            const url = `https://api.themoviedb.org/3/search/keyword?query=${keyword}&page=1`;
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
            const ids = [];

            if (result.total_results > 0) {
                for (const item of result.results) {
                    if (item.name) {
                        ids.push(item.id);
                    }
                }
            }
            return "&with_keywords=" + ids.join("|");
        } catch (error) {
            console.error('Error fetching data:', error);
        }
    }
    return "";
}