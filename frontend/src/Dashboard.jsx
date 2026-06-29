import React, { useState, useRef } from 'react';
import axios from 'axios';
import api from './api'


export default function Dashboard() {
  const [file, setFile] = useState(null);
  const [isDragging, setIsDragging] = useState(false);
  const [uploadState, setUploadState] = useState('idle'); // 'idle', 'uploading', 'success', 'error'
  const fileInputRef = useRef(null);

  // --- DRAG AND DROP HANDLERS ---
  const handleDragOver = (e) => {
    e.preventDefault();
    setIsDragging(true);
  };

  const handleDragLeave = () => {
    setIsDragging(false);
  };

  const handleDrop = (e) => {
    e.preventDefault();
    setIsDragging(false);
    
    if (e.dataTransfer.files && e.dataTransfer.files.length > 0) {
      const droppedFile = e.dataTransfer.files[0];
      if (droppedFile.type === 'application/pdf') {
        setFile(droppedFile);
      } else {
        alert("Please upload a valid PDF file.");
      }
    }
  };

  const handleFileSelect = (e) => {
    if (e.target.files && e.target.files.length > 0) {
      setFile(e.target.files[0]);
    }
  };

  // --- UPLOAD TO BACKEND ---
  const handleUpload = async () => {
    if (!file) {
      alert("Please select a file first.");
      return;
    }

    setUploadState('uploading');

    // 1. Files require FormData, NOT standard JSON!
    const formData = new FormData();
    formData.append('file', file);

    try {
      // 2. Send it using your pre-configured axios interceptor
      // Note: We don't need to pass the token, the interceptor does it automatically!
      // Axios also automatically sets the 'multipart/form-data' boundary for us.
      const response = await api.post('/load_kb', formData);
      
      if (response.status === 200) {
        setUploadState('success');
        alert("File uploaded successfully! Backend processing started.");
        setFile(null); // Reset form
      }
    } catch (error) {
      console.error("Upload failed:", error);
      setUploadState('error');
      alert("Failed to upload the file. Please check your backend.");
    } finally {
      if (uploadState !== 'error') {
        setTimeout(() => setUploadState('idle'), 3000);
      }
    }
  };

  return (
    <div className="min-h-screen bg-slate-50 flex items-center justify-center p-4 font-sans text-slate-800">
      
      {/* Upload Modal/Card */}
      <div className="w-full max-w-lg bg-white rounded-xl shadow-lg border border-slate-200 overflow-hidden relative">
        
        {/* Header */}
        <div className="px-6 py-4 border-b border-slate-100 flex justify-between items-center">
          <h2 className="text-lg font-bold text-slate-800">Add Knowledge Base</h2>
          <button className="text-slate-400 hover:text-slate-600">
            {/* Simple X icon */}
            <svg className="w-5 h-5" fill="none" stroke="currentColor" viewBox="0 0 24 24"><path strokeLinecap="round" strokeLinejoin="round" strokeWidth="2" d="M6 18L18 6M6 6l12 12"></path></svg>
          </button>
        </div>

        <div className="p-6 space-y-6">
          
          {/* Tabs */}
          <div className="flex bg-slate-100/50 p-1 rounded-lg">
            <button className="flex-1 py-2 bg-white shadow-sm rounded-md text-sm font-medium text-blue-700">
              Upload PDF
            </button>
            <button className="flex-1 py-2 text-sm font-medium text-slate-500 hover:text-slate-700">
              Add URL
            </button>
          </div>

          {/* Drag & Drop Zone */}
          <div 
            className={`relative border-2 border-dashed rounded-xl p-8 text-center transition-all duration-200 ease-in-out cursor-pointer
              ${isDragging ? 'border-blue-500 bg-blue-50' : 'border-slate-300 hover:border-slate-400 hover:bg-slate-50'}
              ${file ? 'border-green-400 bg-green-50' : ''}
            `}
            onDragOver={handleDragOver}
            onDragLeave={handleDragLeave}
            onDrop={handleDrop}
            onClick={() => fileInputRef.current?.click()}
          >
            <input 
              type="file" 
              accept=".pdf" 
              className="hidden" 
              ref={fileInputRef}
              onChange={handleFileSelect}
            />
            
            <div className="flex flex-col items-center justify-center space-y-3">
              {file ? (
                // File Selected State
                <>
                  <div className="p-3 bg-green-100 text-green-600 rounded-full">
                    <svg className="w-8 h-8" fill="none" stroke="currentColor" viewBox="0 0 24 24"><path strokeLinecap="round" strokeLinejoin="round" strokeWidth="2" d="M9 12l2 2 4-4m6 2a9 9 0 11-18 0 9 9 0 0118 0z"></path></svg>
                  </div>
                  <div className="text-sm font-medium text-slate-700">{file.name}</div>
                  <div className="text-xs text-slate-500">{(file.size / 1024 / 1024).toFixed(2)} MB</div>
                </>
              ) : (
                // Empty State
                <>
                  <div className="p-3 bg-blue-50 text-blue-600 rounded-full">
                    <svg className="w-8 h-8" fill="none" stroke="currentColor" viewBox="0 0 24 24"><path strokeLinecap="round" strokeLinejoin="round" strokeWidth="2" d="M7 16a4 4 0 01-.88-7.903A5 5 0 1115.9 6L16 6a5 5 0 011 9.9M15 13l-3-3m0 0l-3 3m3-3v12"></path></svg>
                  </div>
                  <p className="text-sm text-slate-600">
                    Drag and drop your PDF here, or <span className="text-blue-600 font-semibold">click to browse</span>
                  </p>
                  <p className="text-xs text-slate-400">Only .pdf files are supported</p>
                </>
              )}
            </div>
          </div>

        </div>

        {/* Footer Actions */}
        <div className="px-6 py-4 bg-slate-50 border-t border-slate-100 flex justify-end">
          <button 
            onClick={handleUpload}
            disabled={!file || uploadState === 'uploading'}
            className={`px-6 py-2.5 rounded-lg text-sm font-semibold text-white transition-all
              ${!file ? 'bg-blue-300 cursor-not-allowed' : 'bg-blue-600 hover:bg-blue-700 shadow-sm'}
              ${uploadState === 'uploading' ? 'opacity-75 cursor-wait' : ''}
            `}
          >
            {uploadState === 'uploading' ? 'Uploading & Processing...' : 'Upload PDF'}
          </button>
        </div>

      </div>
    </div>
  );
}