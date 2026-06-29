import axios from 'axios';

// 1. Create a custom Axios instance
const api = axios.create({
  baseURL: 'http://localhost:8000', // Your Python backend URL
});

// 2. Set up the Interceptor (The Mailroom Assistant)
api.interceptors.request.use(
  (config) => {
    // Look in the browser's safe for the token
    const token = localStorage.getItem('token');
    
    // If the token exists, attach it to the headers
    if (token) {
      config.headers.Authorization = `Bearer ${token}`;
    }
    
    return config; // Send the request on its way!
  },
  (error) => {
    return Promise.reject(error);
  }
);

api.interceptors.response.use(
  (response) => {
    // If the request was successful, just pass the data through normally
    return response; 
  },
  (error) => {
    // If Python sends back an error, check if it's a 401 (Expired/Invalid Token)
    if (error.response && error.response.status === 401) {
      console.warn("Session expired. Logging out user...");
      
      // 1. Delete the dead token from the browser's safe
      localStorage.removeItem('token');
      
      // 2. Force the browser to redirect back to the Auth page
      window.location.href = '/auth'; 
    }
    
    // Pass the error along just in case you want to handle it elsewhere
    return Promise.reject(error);
  }
);

export default api;