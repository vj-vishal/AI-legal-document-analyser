import React from 'react';
import { BrowserRouter as Router, Routes, Route, Navigate } from 'react-router-dom';
import LandingPage from './LandingPage';
import Auth from './Auth';
import Dashboard from './Dashboard';
import MasterDashboard from './MasterDashboard';

export default function App() {
  return (
    <Router>
      <Routes>
        <Route path="/" element={<LandingPage />} />
        <Route path="/auth" element={<Auth />} />
        <Route path="/dashboard" element={<Dashboard />} />
        <Route path="/masterdashboard" element={<MasterDashboard />} />
      </Routes>
    </Router>
  );
}