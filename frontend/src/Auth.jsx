import React, { useState } from 'react';
import { useNavigate, useLocation } from 'react-router-dom';

export default function Auth() {
  const navigate = useNavigate();
  const location = useLocation(); // 2. Read the incoming message

  // 3. Set the default state based on which button they clicked!
  // If the message says 'signup', start with isLogin as false. Otherwise, default to true.
  const [isLogin, setIsLogin] = useState(location.state?.mode === 'signup' ? false : true);
  
  // State for our form fields
  const [name, setName] = useState('');
  const [email, setEmail] = useState('');
  const [password, setPassword] = useState('');
  

  // Handle switching between Log In and Sign Up gracefully
  const toggleAuthMode = () => {
    setIsLogin(!isLogin);
    // Clear out the fields when switching modes
    setName('');
    setEmail('');
    setPassword('');
  };

  const handleSubmit = async (e) => {
    e.preventDefault(); 

    // 1. DYNAMIC PAYLOAD: Construct the exact JSON your Python backend needs
    const payload = isLogin 
      ? { email: email, password: password } // Log In payload
      : { name: name, email: email, password: password }; // Sign Up payload

    try {
      // 2. DYNAMIC ENDPOINT: Route to the correct Python URL
      const endpoint = isLogin 
        ? 'http://localhost:8000/login' 
        : 'http://localhost:8000/signup';
      
      const response = await fetch(endpoint, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify(payload)
      });

      if (response.ok) {
        const data = await response.json();
        // Save the secure token (Assuming your backend sends {'access_token': '...'})
        if (data.access_token) {
          localStorage.setItem('token', data.access_token);
        }
        alert(`Success! You have successfully ${isLogin ? 'logged in' : 'signed up'}.`);
        navigate('/masterdashboard'); 
      } else {
        alert("Authentication failed. Please check your credentials or backend.");
      }
    } catch (error) {
      console.error("Failed to connect to backend:", error);
      alert("Could not reach the Python server. Make sure your FastAPI/Flask server is running.");
    }
  };

  return (
    <div className="min-h-screen flex items-center justify-center bg-slate-50 py-12 px-4 sm:px-6 lg:px-8">
      <div className="max-w-md w-full space-y-8 bg-white p-10 border border-slate-200 rounded-xl shadow-sm">
        
        {/* Header */}
        <div className="text-center">
          <h2 className="text-3xl font-extrabold text-slate-900">
            {isLogin ? 'Sign in to your account' : 'Create your workspace'}
          </h2>
          <p className="mt-2 text-sm text-slate-600">
            {isLogin ? "Don't have an account? " : "Already have an account? "}
            <button 
              type="button"
              onClick={toggleAuthMode} 
              className="font-medium text-blue-700 hover:text-blue-600 transition-colors"
            >
              {isLogin ? 'Sign up here' : 'Log in here'}
            </button>
          </p>
        </div>

        {/* Form */}
        <form className="mt-8 space-y-6" onSubmit={handleSubmit}>
          <div className="space-y-4 rounded-md shadow-sm">
            
            {/* CONDITIONAL RENDERING: Only show Name field if NOT logging in */}
            {!isLogin && (
              <div>
                <label className="block text-sm font-medium text-slate-700 mb-1">Full Name</label>
                <input 
                  type="text" 
                  required={!isLogin} // Only require it if signing up
                  value={name} 
                  onChange={(e) => setName(e.target.value)}
                  className="appearance-none relative block w-full px-3 py-2 border border-slate-300 rounded-md focus:outline-none focus:ring-blue-500 focus:border-blue-500 sm:text-sm" 
                  placeholder="Jane Doe"
                />
              </div>
            )}

            <div>
              <label className="block text-sm font-medium text-slate-700 mb-1">Email address</label>
              <input 
                type="email" 
                required 
                value={email} 
                onChange={(e) => setEmail(e.target.value)}
                className="appearance-none relative block w-full px-3 py-2 border border-slate-300 rounded-md focus:outline-none focus:ring-blue-500 focus:border-blue-500 sm:text-sm" 
                placeholder="name@company.com"
              />
            </div>
            
            <div>
              <label className="block text-sm font-medium text-slate-700 mb-1">Password</label>
              <input 
                type="password" 
                required 
                value={password} 
                onChange={(e) => setPassword(e.target.value)}
                className="appearance-none relative block w-full px-3 py-2 border border-slate-300 rounded-md focus:outline-none focus:ring-blue-500 focus:border-blue-500 sm:text-sm" 
                placeholder="••••••••"
              />
            </div>
          </div>

          <button 
            type="submit" 
            className="w-full flex justify-center py-2.5 px-4 border border-transparent text-sm font-medium rounded-lg text-white bg-blue-700 hover:bg-blue-800 focus:outline-none transition-colors shadow-sm"
          >
            {isLogin ? 'Sign In' : 'Create Account'}
          </button>
        </form>
      </div>
    </div>
  );
}