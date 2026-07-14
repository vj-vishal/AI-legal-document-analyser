import React from 'react';
import { BrowserRouter as Router, Routes, Route, Navigate, Outlet } from 'react-router-dom';
import LandingPage from './LandingPage';
import Auth from './Auth';
import Dashboard from './Dashboard';
import MasterDashboard from './MasterDashboard';

// ==========================================
// 1. PUBLIC ROUTE GUARD
// Kicks logged-in users out of Auth and Landing pages
// ==========================================
const PublicRoute = () => {
  const token = localStorage.getItem('token');
  if (token) {
    return <Navigate to="/masterdashboard" replace />;
  }
  return <Outlet />;
};

// ==========================================
// 2. PROTECTED ROUTE GUARD
// Kicks logged-out users out of the Dashboard
// ==========================================
const ProtectedRoute = () => {
  const token = localStorage.getItem('token');
  if (!token) {
    return <Navigate to="/auth" replace />;
  }
  return <Outlet />;
};

export default function App() {
  return (
    <Router>
      <Routes>
        
        {/* --- PUBLIC ZONE (Blocked if Logged In) --- */}
        <Route element={<PublicRoute />}>
          <Route path="/" element={<LandingPage />} />
          <Route path="/auth" element={<Auth />} />
        </Route>

        {/* --- PROTECTED ZONE (Blocked if Logged Out) --- */}
        <Route element={<ProtectedRoute />}>
          {/* <Route path="/dashboard" element={<Dashboard />} /> */}
          <Route path="/masterdashboard" element={<MasterDashboard />} />
        </Route>

      </Routes>
    </Router>
  );
}


// export default function App() {
//   return (
//     <Router>
//       <Routes>
//         <Route path="/" element={<LandingPage />} />
//         <Route path="/auth" element={<Auth />} />
//         {/* <Route path="/dashboard" element={<Dashboard />} /> */}
//         <Route path="/masterdashboard" element={<MasterDashboard />} />
//       </Routes>
//     </Router>
//   );
// }