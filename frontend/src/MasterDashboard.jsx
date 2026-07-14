import React, { useState, useRef, useEffect } from 'react';
import { useNavigate } from 'react-router-dom'; // NEW: Import for secure routing
import axios from 'axios';
import api from './api';

export default function Dashboard() {
  const navigate = useNavigate(); // NEW: Hook for routing

  // --- USER STATE ---
  const [userName, setUserName] = useState(''); 
  const [creditsLeft, setCreditsLeft] = useState('...'); 

  // --- NAVIGATION STATE ---
  const [activeView, setActiveView] = useState('home'); 
  const [showSettingsMenu, setShowSettingsMenu] = useState(false); // NEW: State for settings popover

  // --- DOCUMENTS DATA STATE ---
  const [documents, setDocuments] = useState([]);
  const [isLoadingDocs, setIsLoadingDocs] = useState(false);

  // --- CHAT STATE ---
  const [chatMessages, setChatMessages] = useState([]);
  const [chatInput, setChatInput] = useState('');
  const [selectedDocId, setSelectedDocId] = useState('');
  const [selectedKbId, setSelectedKbId] = useState('');
  const [sessionId, setSessionId] = useState(null);
  const [isChatLoading, setIsChatLoading] = useState(false);
  const [chatSessions, setChatSessions] = useState([]); 
  const chatScrollRef = useRef(null);

  // --- LOGOUT HANDLER --- // NEW
  const handleLogout = () => {
    // 1. Clear authentication tokens (adjust 'token' to whatever key you use)
    localStorage.removeItem('token'); 
    
    // 2. Redirect to login and REPLACE history so they cannot click 'Back' to return here
    navigate('/', { replace: true });
  };

  // Auto-scroll chat to bottom when new messages arrive
  useEffect(() => {
    if (chatScrollRef.current) {
      chatScrollRef.current.scrollTop = chatScrollRef.current.scrollHeight;
    }
  }, [chatMessages]);

  // --- FETCH RECENT CHATS EFFECT ---
  const fetchChatSessions = async () => {
    try {
      const response = await api.get('/chat_session_view');
      if (response.status === 200) {
        setChatSessions(response.data.data || response.data); 
      }
    } catch (error) {
      console.error("Failed to fetch chat sessions:", error);
    }
  };

  // Fetch initial data on mount
  useEffect(() => {
    // Fetch User Profile
    const fetchProfile = async () => {
      try {
        const response = await api.get('/user_profile'); 
        if (response.status === 200) {
          const fetchedName = response.data.data[0]?.name;
          setUserName(fetchedName || 'User'); 
        }
      } catch (error) {
        console.error("Failed to fetch profile:", error);
        setUserName('User'); 
      }
    };

    const fetchCredits = async () => {
      try {
        const response = await api.get('/rate_limit_status'); 
        if (response.status === 200) {
          setCreditsLeft(response.data.remaining_credits ?? '...');
        }
      } catch (error) {
        console.error("Failed to fetch credits:", error);
        setCreditsLeft('...'); 
      }
    };

    fetchProfile();
    fetchChatSessions();
    fetchCredits(); 
  }, []);

  // --- FETCH DOCUMENTS EFFECT ---
  useEffect(() => {
    if (activeView === 'documents' || activeView === 'chat') {
      const fetchDocuments = async () => {
        setIsLoadingDocs(true);
        try {
          const response = await api.get('/user_kb_docs');
          if (response.status === 200) {
            setDocuments(response.data.data);
          }
        } catch (error) {
          console.error("Failed to fetch documents:", error);
        } finally {
          setIsLoadingDocs(false);
        }
      };

      fetchDocuments();
    }
  }, [activeView]);

  // --- NEW: POLLING EFFECT FOR BACKGROUND PROCESSING ---
  useEffect(() => {
    // Check if any document in our state currently has a 'processing' status
    const hasProcessingDocs = documents.some(doc => doc.status === 'processing' || doc.status === 'pending');
    
    let pollInterval;

    // If there are processing documents, set up a timer to ask the backend for updates
    if (hasProcessingDocs) {
      pollInterval = setInterval(async () => {
        try {
          const response = await api.get('/user_kb_docs');
          if (response.status === 200) {
            setDocuments(response.data.data);
          }
        } catch (error) {
          console.error("Polling failed:", error);
        }
      }, 3000); // Check every 3 seconds
    }

    // Cleanup: Stop the timer if the component unmounts or if all documents finish processing
    return () => {
      if (pollInterval) clearInterval(pollInterval);
    };
  }, [documents]); // This effect re-runs every time the documents array updates

  // --- UPLOAD STATE ---
  const [showUploadModal, setShowUploadModal] = useState(false);
  const [file, setFile] = useState(null);
  const [isDragging, setIsDragging] = useState(false);
  const [uploadState, setUploadState] = useState('idle');
  const fileInputRef = useRef(null);

  // --- DRAG AND DROP HANDLERS ---
  const handleDragOver = (e) => { e.preventDefault(); setIsDragging(true); };
  const handleDragLeave = () => { setIsDragging(false); };
  const handleDrop = (e) => {
    e.preventDefault();
    setIsDragging(false);
    if (e.dataTransfer.files && e.dataTransfer.files.length > 0) {
      const droppedFile = e.dataTransfer.files[0];
      if (droppedFile.type === 'application/pdf') setFile(droppedFile);
      else alert("Please upload a valid PDF file.");
    }
  };
  const handleFileSelect = (e) => {
    if (e.target.files && e.target.files.length > 0) setFile(e.target.files[0]);
  };

  // --- UPLOAD TO BACKEND ---
  const handleUpload = async () => {
    if (!file) return alert("Please select a file first.");
    setUploadState('uploading');
    
    const formData = new FormData();
    formData.append('file', file);

    let isError = false;

    try {
      const response = await api.post('/load_kb', formData);
      
      // UPDATED: Accept 202 status code since Celery returns accepted, not instantly finished
      if (response.status === 200 || response.status === 202) {
        setUploadState('success');
        alert("File uploaded successfully! Backend processing started.");
        setFile(null);
        setTimeout(() => setShowUploadModal(false), 1500); 
        
        // UPDATED: Always fetch docs immediately after upload success to inject the 'processing' 
        // document into state, which automatically triggers our new polling useEffect above.
        const res = await api.get('/user_kb_docs');
        if (res.status === 200) {
          setDocuments(res.data.data);
        }
      }
    } catch (error) {
    console.error("Upload failed:", error);
    setUploadState('error');
    isError = true; 

    if (error.response) {
      const statusCode = error.response.status;
      const responseData = error.response.data;

      switch (statusCode) {
        case 400: {
          const typeMessage = responseData?.detail || "Only PDF files are allowed.";
          alert(`File Type Error: ${typeMessage}`);
          break;
        }
        case 413: {
          const sizeMessage = responseData?.detail || "File exceeds the maximum allowed size.";
          alert(`File Size Error: ${sizeMessage}`);
          break;
        }
        case 422: {
          let validationMessage = "Validation failed.";
          if (responseData && responseData.detail) {
            if (typeof responseData.detail === 'string') {
              validationMessage = responseData.detail;
            } else if (Array.isArray(responseData.detail)) {
              validationMessage = responseData.detail[0]?.msg || "Invalid input data format.";
            }
          }
          alert(`Validation Error: ${validationMessage}`);
          break;
        }
        case 429: {
          const rateLimitMessage = responseData?.detail || "Rate limit exceeded. Please try again later.";
          alert(`Upload blocked: ${rateLimitMessage}`);
          break;
        }
        default: {
          const serverMessage = responseData?.detail || "Failed to upload the file. Please check your backend.";
          alert(`Server Error (${statusCode}): ${serverMessage}`);
          break;
        }
      }
    } else {
      alert("Network error: Cannot reach the backend server. Please verify it is running.");
    }
  } finally {
    if (!isError) {
      setTimeout(() => setUploadState('idle'), 3000);
    }
  }
};

  // --- CHAT HANDLERS ---
  const handleDocSelection = (e) => {
    const docId = e.target.value;
    setSelectedDocId(docId);
    
    const docObj = documents.find(d => d.document_id === docId);
    if (docObj) {
      setSelectedKbId(docObj.knowledge_base_id);
    } else {
      setSelectedKbId('');
    }
  };

  const loadChatSession = async (session) => {
    setActiveView('chat');
    setSessionId(session.id);
    if (session.knowledge_base_id) setSelectedKbId(session.knowledge_base_id);
    
    setChatMessages([]); 
    setIsChatLoading(true);
    
    try {
      const response = await api.get('/chat_message_view', {
        params: { session_id: session.id }
      });
      
      if (response.status === 200 && response.data.data) {
        const history = response.data.data.map(msg => ({
          role: msg.role,
          content: msg.messages 
        }));
        setChatMessages(history);
      }
    } catch (error) {
      console.error("Failed to load chat history:", error);
      alert("Failed to load conversation history.");
    } finally {
      setIsChatLoading(false);
    }
  };

  const handleSendMessage = async () => {
    if (!chatInput.trim()) return;

    const userMsg = { role: 'user', content: chatInput };
    setChatMessages(prev => [...prev, userMsg]);
    setChatInput('');
    setIsChatLoading(true);

    const payload = {
      query: userMsg.content,
      knowledge_base_id: selectedKbId || null,
      kb_document_id: selectedDocId || null
    };
    
    if (sessionId) {
      payload.session_id = sessionId;
    }

    try {
      const response = await api.post('/chat', payload);

      if (response.status === 200) {
        const botMsg = { role: 'assistant', content: response.data.answer || response.data.message };
        setChatMessages(prev => [...prev, botMsg]);
        
        if (response.data.day_adjusted_token !== undefined) {
          setCreditsLeft(Math.max(0, response.data.day_adjusted_token));
        }

        if (response.data.session_id) {
          if (!sessionId) {
            setSessionId(response.data.session_id);
            fetchChatSessions(); 
          } else {
            setSessionId(response.data.session_id);
          }
        }
      }
    } catch (error) {
      console.error("Chat error:", error);

      if (error.response && error.response.status === 429) {
        setCreditsLeft(0);
        const errorMessage = error.response.data?.detail || "Rate limit exceeded. Please try again later.";
        setChatMessages(prev => [...prev, { 
          role: 'assistant', 
          content: `Error: ${errorMessage}` 
        }]);
      } else {
      setChatMessages(prev => [...prev, { role: 'assistant', content: "Error: Could not connect to the chat service. Please try again." }]);
      }
    } finally {
      setIsChatLoading(false);
    }
  };

  return (
    <div className="flex h-screen bg-slate-50 text-slate-900 font-sans overflow-hidden selection:bg-blue-200">
      
      {/* ================= SIDEBAR ================= */}
      <aside className="w-64 bg-white border-r border-slate-200 flex flex-col">
        <div className="p-6 flex items-center gap-3 border-b border-slate-100">
          <div className="w-8 h-8 bg-blue-700 text-white flex items-center justify-center rounded-lg font-bold text-xl shadow-sm">
            ⚖️
          </div>
          <div>
            <h1 className="text-xl font-bold text-slate-900 tracking-tight leading-tight">ApnaKanoon</h1>
            <p className="text-[10px] text-blue-700 font-bold uppercase tracking-wider">Legal Intelligence</p>
          </div>
        </div>

        <nav className="flex-1 px-4 py-6 space-y-1 flex flex-col overflow-hidden">
          
          <div className="shrink-0 space-y-1">
            <button 
              onClick={() => setActiveView('home')}
              className={`w-full flex items-center gap-3 px-3 py-2.5 rounded-lg font-semibold text-sm transition-colors border ${
                activeView === 'home' 
                  ? 'bg-blue-50 text-blue-700 border-blue-100/50' 
                  : 'text-slate-600 hover:text-blue-700 hover:bg-slate-50 border-transparent'
              }`}
            >
              <svg className="w-5 h-5" fill="none" stroke="currentColor" viewBox="0 0 24 24"><path strokeLinecap="round" strokeLinejoin="round" strokeWidth="2" d="M4 6a2 2 0 012-2h2a2 2 0 012 2v2a2 2 0 01-2 2H6a2 2 0 01-2-2V6zM14 6a2 2 0 012-2h2a2 2 0 012 2v2a2 2 0 01-2 2h-2a2 2 0 01-2-2V6zM4 16a2 2 0 012-2h2a2 2 0 012 2v2a2 2 0 01-2 2H6a2 2 0 01-2-2v-2zM14 16a2 2 0 012-2h2a2 2 0 012 2v2a2 2 0 01-2 2h-2a2 2 0 01-2-2v-2z"></path></svg>
              Dashboard
            </button>
            
            <button 
              onClick={() => {
                setActiveView('chat');
                setChatMessages([]);
                setSessionId(null);
                setSelectedDocId('');
                setSelectedKbId('');
              }}
              className={`w-full flex items-center gap-3 px-3 py-2.5 rounded-lg font-semibold text-sm transition-colors border ${
                activeView === 'chat' && !sessionId
                  ? 'bg-blue-50 text-blue-700 border-blue-100/50' 
                  : 'text-slate-600 hover:text-blue-700 hover:bg-slate-50 border-transparent'
              }`}
            >
              <svg className="w-5 h-5" fill="none" stroke="currentColor" viewBox="0 0 24 24"><path strokeLinecap="round" strokeLinejoin="round" strokeWidth="2" d="M8 12h.01M12 12h.01M16 12h.01M21 12c0 4.418-4.03 8-9 8a9.863 9.863 0 01-4.255-.949L3 20l1.395-3.72C3.512 15.042 3 13.574 3 12c0-4.418 4.03-8 9-8s9 3.582 9 8z"></path></svg>
              New Chat
            </button>

            <button 
              onClick={() => setActiveView('documents')}
              className={`w-full flex items-center gap-3 px-3 py-2 rounded-lg text-sm font-medium transition-colors border ${
                activeView === 'documents'
                  ? 'bg-blue-50 text-blue-700 border-blue-100/50'
                  : 'text-slate-600 hover:text-blue-700 hover:bg-slate-50 border-transparent'
              }`}
            >
              <svg className="w-5 h-5" fill="none" stroke="currentColor" viewBox="0 0 24 24"><path strokeLinecap="round" strokeLinejoin="round" strokeWidth="2" d="M9 12h6m-6 4h6m2 5H7a2 2 0 01-2-2V5a2 2 0 012-2h5.586a1 1 0 01.707.293l5.414 5.414a1 1 0 01.293.707V19a2 2 0 01-2 2z"></path></svg>
              Documents
            </button>
          </div>

          {/* DYNAMIC RECENT CHATS LIST */}
          <div className="pt-6 pb-2 px-3 text-xs font-bold text-slate-400 uppercase tracking-wider shrink-0">Recent Chats</div>
          
          <div className="flex-1 overflow-y-auto space-y-1 pr-1 custom-scrollbar">
            {chatSessions.length > 0 ? (
              chatSessions.slice().reverse().map((session, idx) => (
                <button 
                  key={session.id || idx}
                  onClick={() => loadChatSession(session)} 
                  className={`w-full flex items-center gap-3 px-3 py-2 rounded-lg text-sm font-medium transition-colors border ${
                    sessionId === session.id && activeView === 'chat'
                      ? 'bg-blue-50 text-blue-700 border-blue-100/50'
                      : 'text-slate-600 hover:text-blue-700 hover:bg-slate-50 border-transparent'
                  }`}
                  title={session.title}
                >
                  <svg className="w-4 h-4 shrink-0 text-slate-400" fill="none" stroke="currentColor" viewBox="0 0 24 24"><path strokeLinecap="round" strokeLinejoin="round" strokeWidth="2" d="M8 10h.01M12 10h.01M16 10h.01M9 16H5a2 2 0 01-2-2V6a2 2 0 012-2h14a2 2 0 012 2v8a2 2 0 01-2 2h-5l-5 5v-5z"></path></svg>
                  <span className="truncate text-left flex-1">{session.title || 'Untitled Chat'}</span>
                </button>
              ))
            ) : (
              <div className="px-3 py-2 text-xs text-slate-400 italic">No recent chats</div>
            )}
          </div>
        </nav>

        {/* ================= BOTTOM SIDEBAR (SETTINGS/LOGOUT) ================= */}
        <div className="p-4 border-t border-slate-200 shrink-0 relative">
          
          {/* NEW: Settings Popover Menu */}
          {showSettingsMenu && (
            <div className="absolute bottom-16 left-4 right-4 bg-white border border-slate-200 rounded-lg shadow-lg overflow-hidden z-50 animate-in fade-in slide-in-from-bottom-2 duration-200">
              <button 
                onClick={handleLogout}
                className="w-full flex items-center gap-3 px-4 py-3 text-red-600 hover:bg-red-50 text-sm font-semibold transition-colors text-left"
              >
                <svg className="w-5 h-5" fill="none" stroke="currentColor" viewBox="0 0 24 24"><path strokeLinecap="round" strokeLinejoin="round" strokeWidth="2" d="M17 16l4-4m0 0l-4-4m4 4H7m6 4v1a3 3 0 01-3 3H6a3 3 0 01-3-3V7a3 3 0 013-3h4a3 3 0 013 3v1"></path></svg>
                Log Out
              </button>
            </div>
          )}

          {/* UPDATED: Settings Toggle Button */}
          <button 
            onClick={() => setShowSettingsMenu(!showSettingsMenu)}
            className={`w-full flex items-center gap-3 px-3 py-2 rounded-lg text-sm font-medium transition-colors border border-transparent
              ${showSettingsMenu ? 'bg-slate-100 text-slate-900' : 'text-slate-600 hover:text-blue-700 hover:bg-slate-50'}
            `}
          >
            <svg className="w-5 h-5" fill="none" stroke="currentColor" viewBox="0 0 24 24"><path strokeLinecap="round" strokeLinejoin="round" strokeWidth="2" d="M10.325 4.317c.426-1.756 2.924-1.756 3.35 0a1.724 1.724 0 002.573 1.066c1.543-.94 3.31.826 2.37 2.37a1.724 1.724 0 001.065 2.572c1.756.426 1.756 2.924 0 3.35a1.724 1.724 0 00-1.066 2.573c.94 1.543-.826 3.31-2.37 2.37a1.724 1.724 0 00-2.572 1.065c-.426 1.756-2.924 1.756-3.35 0a1.724 1.724 0 00-2.573-1.066c-1.543.94-3.31-.826-2.37-2.37a1.724 1.724 0 00-1.065-2.572c-1.756-.426-1.756-2.924 0-3.35a1.724 1.724 0 001.066-2.573c-.94-1.543.826-3.31 2.37-2.37.996.608 2.296.07 2.572-1.065z"></path><path strokeLinecap="round" strokeLinejoin="round" strokeWidth="2" d="M15 12a3 3 0 11-6 0 3 3 0 016 0z"></path></svg>
            Settings
          </button>
        </div>
      </aside>

      {/* ================= MAIN CONTENT ================= */}
      <main className="flex-1 flex flex-col min-w-0 overflow-hidden">
        
        {/* Top Header */}
        <header className="h-16 border-b border-slate-200 flex items-center justify-between px-8 bg-white/80 backdrop-blur-md">
          <div className="w-96 relative">
            <svg className="w-4 h-4 absolute left-3 top-3 text-slate-400" fill="none" stroke="currentColor" viewBox="0 0 24 24"><path strokeLinecap="round" strokeLinejoin="round" strokeWidth="2" d="M21 21l-6-6m2-5a7 7 0 11-14 0 7 7 0 0114 0z"></path></svg>
            <input 
              type="text" 
              placeholder="Search chats, docs, notes... (Ctrl+K)" 
              className="w-full bg-slate-50 border border-slate-200 rounded-lg pl-10 pr-4 py-2 text-sm text-slate-800 focus:outline-none focus:border-blue-700 focus:bg-white transition-colors"
            />
          </div>
          <div className="flex items-center gap-4">
            <button 
              onClick={() => setShowUploadModal(true)}
              className="flex items-center gap-2 px-4 py-1.5 bg-blue-700 hover:bg-blue-800 text-white rounded-lg text-sm font-semibold transition-all shadow-sm"
            >
              <svg className="w-4 h-4" fill="none" stroke="currentColor" viewBox="0 0 24 24"><path strokeLinecap="round" strokeLinejoin="round" strokeWidth="2" d="M4 16v1a3 3 0 003 3h10a3 3 0 003-3v-1m-4-8l-4-4m0 0L8 8m4-4v12"></path></svg>
              Upload
            </button>
            <div className="flex items-center gap-2 text-sm bg-slate-50 px-3 py-1.5 rounded-full border border-slate-200 font-medium">
              <span className="text-blue-700">🪙</span> {creditsLeft} free left
            </div>
            <div className="w-9 h-9 rounded-full bg-blue-700 text-white font-bold flex items-center justify-center shadow-sm">
              {userName ? userName.charAt(0).toUpperCase() : 'U'}
            </div>
          </div>
        </header>

        {/* Dynamic Body Content */}
        <div className="flex-1 overflow-hidden flex flex-col">
          
          {/* --- HOME VIEW --- */}
          {activeView === 'home' && (
            <div className="flex-1 overflow-y-auto p-8">
              <div className="max-w-6xl mx-auto space-y-8">
                <div className="bg-white border border-slate-200 rounded-2xl p-8 flex items-center justify-between shadow-sm relative overflow-hidden">
                  <div className="absolute top-0 right-0 w-64 h-64 bg-blue-50 rounded-full blur-3xl -mr-20 -mt-20 opacity-50 pointer-events-none"></div>

                  <div className="flex items-center gap-5 relative z-10">
                    <div className="w-16 h-16 rounded-full bg-blue-50 text-blue-700 font-bold text-3xl flex items-center justify-center border border-blue-100">
                      {userName ? userName.charAt(0).toUpperCase() : 'U'}
                    </div>
                    <div>
                      <h2 className="text-2xl font-bold text-slate-900 tracking-tight">Welcome {userName}! <span className="text-xs align-middle bg-slate-100 px-2 py-1 rounded-md text-slate-600 ml-2 font-semibold border border-slate-200">Free Tier</span></h2>
                      <p className="text-slate-600 mt-1 font-medium">Welcome to your ApnaKanoon workspace</p>
                    </div>
                  </div>
                  <div className="flex gap-3 relative z-10">
                    <button 
                      onClick={() => setShowUploadModal(true)}
                      className="flex items-center gap-2 px-5 py-2.5 bg-white hover:bg-slate-50 text-slate-700 rounded-lg font-semibold transition-colors border border-slate-200 shadow-sm"
                    >
                      <svg className="w-4 h-4 text-slate-500" fill="none" stroke="currentColor" viewBox="0 0 24 24"><path strokeLinecap="round" strokeLinejoin="round" strokeWidth="2" d="M4 16v1a3 3 0 003 3h10a3 3 0 003-3v-1m-4-8l-4-4m0 0L8 8m4-4v12"></path></svg>
                      Upload File
                    </button>
                    <button 
                      onClick={() => { setActiveView('chat'); setChatMessages([]); setSessionId(null); setSelectedDocId(''); setSelectedKbId(''); }}
                      className="flex items-center gap-2 px-5 py-2.5 bg-blue-700 hover:bg-blue-800 text-white rounded-lg font-semibold transition-all shadow-sm"
                    >
                      <svg className="w-4 h-4" fill="none" stroke="currentColor" viewBox="0 0 24 24"><path strokeLinecap="round" strokeLinejoin="round" strokeWidth="2" d="M12 4v16m8-8H4"></path></svg>
                      New Chat
                    </button>
                  </div>
                </div>

                <div className="grid grid-cols-1 md:grid-cols-4 gap-4">
                  {[
                    { title: "Total Chats", value: chatSessions.length || "0", icon: "💬", color: "bg-blue-50 text-blue-700 border-blue-100" },
                    { title: "Documents", value: documents.length || "0", icon: "📄", color: "bg-slate-50 text-slate-700 border-slate-200" },
                    { title: "Notes", value: "0", icon: "📝", color: "bg-slate-50 text-slate-700 border-slate-200" },
                    { title: "Storage", value: "0MB/100MB", icon: "💾", color: "bg-slate-50 text-slate-700 border-slate-200" }
                  ].map((stat, i) => (
                    <div key={i} className="bg-white border border-slate-200 p-5 rounded-2xl flex items-center justify-between shadow-sm hover:border-blue-200 transition-colors">
                      <div>
                        <p className="text-slate-500 font-medium text-sm mb-1">{stat.title}</p>
                        <p className="text-2xl font-bold text-slate-900">{stat.value}</p>
                      </div>
                      <div className={`w-12 h-12 rounded-xl flex items-center justify-center border ${stat.color} text-xl`}>
                        {stat.icon}
                      </div>
                    </div>
                  ))}
                </div>

                <div className="bg-blue-50/50 border border-blue-100 rounded-2xl p-5 flex items-start gap-4">
                  <div className="text-blue-700 mt-0.5 text-lg">💡</div>
                  <div>
                    <h4 className="text-blue-900 font-bold text-sm mb-1">Legal Tip of the Day</h4>
                    <p className="text-blue-700/80 text-sm font-medium">Keep records of all communications in legal matters, including emails and messages.</p>
                  </div>
                </div>
              </div>
            </div>
          )}

          {/* --- DOCUMENTS TABLE VIEW --- */}
          {activeView === 'documents' && (
            <div className="flex-1 overflow-y-auto p-8">
              <div className="max-w-6xl mx-auto">
                <div className="bg-white border border-slate-200 rounded-2xl shadow-sm overflow-hidden">
                  <div className="px-6 py-5 border-b border-slate-100 flex justify-between items-center bg-slate-50/50">
                    <div>
                      <h3 className="text-lg font-bold text-slate-900">Your Knowledge Base</h3>
                      <p className="text-sm text-slate-500 mt-1">Manage and view your uploaded legal documents.</p>
                    </div>
                  </div>
                  
                  <div className="p-6">
                    {isLoadingDocs ? (
                      <div className="flex flex-col items-center justify-center py-12 text-slate-500">
                        <svg className="animate-spin h-8 w-8 text-blue-700 mb-4" xmlns="http://www.w3.org/2000/svg" fill="none" viewBox="0 0 24 24"><circle className="opacity-25" cx="12" cy="12" r="10" stroke="currentColor" strokeWidth="4"></circle><path className="opacity-75" fill="currentColor" d="M4 12a8 8 0 018-8V0C5.373 0 0 5.373 0 12h4zm2 5.291A7.962 7.962 0 014 12H0c0 3.042 1.135 5.824 3 7.938l3-2.647z"></path></svg>
                        <p className="font-medium">Loading your documents...</p>
                      </div>
                    ) : documents.length > 0 ? (
                      <div className="overflow-x-auto">
                        <table className="w-full text-left border-collapse">
                          <thead>
                            <tr className="border-b border-slate-200 text-sm font-semibold text-slate-600 bg-slate-50">
                              <th className="p-4 rounded-tl-lg">Doc Id</th>
                              <th className="p-4">Document Name</th>
                              <th className="p-4">Type</th>
                              <th className="p-4">Upload Date</th>
                              <th className="p-4 rounded-tr-lg">Status</th>
                            </tr>
                          </thead>
                          <tbody className="text-sm">
                            {documents.map((doc, idx) => (
                              <tr key={idx} className="border-b border-slate-100 hover:bg-slate-50/80 transition-colors">
                                <td className="p-4 text-slate-500">{doc.document_id || 'None'}</td>  
                                <td className="p-4 font-medium text-slate-900 flex items-center gap-3">
                                  <svg className="w-5 h-5 text-emerald-600" fill="none" stroke="currentColor" viewBox="0 0 24 24"><path strokeLinecap="round" strokeLinejoin="round" strokeWidth="2" d="M9 12h6m-6 4h6m2 5H7a2 2 0 01-2-2V5a2 2 0 012-2h5.586a1 1 0 01.707.293l5.414 5.414a1 1 0 01.293.707V19a2 2 0 01-2 2z"></path></svg>
                                  {doc.title}
                                </td>
                                <td className="p-4 text-slate-500">{doc.document_type || 'PDF'}</td>
                                <td className="p-4 text-slate-500">
                                  {doc.created_at ? new Date(doc.created_at).toLocaleDateString() : 'N/A'}
                                </td>
                                <td className="p-4">
                                  <span className={`px-2.5 py-1 border rounded-md text-xs font-semibold capitalize ${
                                    doc.status === 'processing' || doc.status === 'pending'
                                      ? 'bg-amber-50 text-amber-700 border-amber-200' 
                                      : doc.status === 'failed'
                                      ? 'bg-red-50 text-red-700 border-red-200'
                                      : 'bg-emerald-50 text-emerald-700 border-emerald-200'
                                  }`}>
                                    {doc.status === 'processing' || doc.status === 'pending' ? (
                                      <span className="flex items-center gap-1">
                                        <svg className="animate-spin h-3 w-3" xmlns="http://www.w3.org/2000/svg" fill="none" viewBox="0 0 24 24"><circle className="opacity-25" cx="12" cy="12" r="10" stroke="currentColor" strokeWidth="4"></circle><path className="opacity-75" fill="currentColor" d="M4 12a8 8 0 018-8V0C5.373 0 0 5.373 0 12h4zm2 5.291A7.962 7.962 0 014 12H0c0 3.042 1.135 5.824 3 7.938l3-2.647z"></path></svg>
                                        Processing
                                      </span>
                                    ) : (
                                      doc.status || 'Processed'
                                    )}
                                  </span>
                                </td>
                              </tr>
                            ))}
                          </tbody>
                        </table>
                      </div>
                    ) : (
                      <div className="text-center py-12">
                        <div className="w-16 h-16 bg-slate-100 text-slate-400 rounded-full flex items-center justify-center mx-auto mb-4">
                          <svg className="w-8 h-8" fill="none" stroke="currentColor" viewBox="0 0 24 24"><path strokeLinecap="round" strokeLinejoin="round" strokeWidth="2" d="M9 12h6m-6 4h6m2 5H7a2 2 0 01-2-2V5a2 2 0 012-2h5.586a1 1 0 01.707.293l5.414 5.414a1 1 0 01.293.707V19a2 2 0 01-2 2z"></path></svg>
                        </div>
                        <h4 className="text-lg font-bold text-slate-900 mb-1">No documents found</h4>
                        <p className="text-slate-500 text-sm mb-6">Upload a PDF to start building your knowledge base.</p>
                        <button 
                          onClick={() => setShowUploadModal(true)}
                          className="px-5 py-2.5 bg-blue-700 text-white rounded-lg text-sm font-semibold hover:bg-blue-800 transition-colors shadow-sm"
                        >
                          Upload First Document
                        </button>
                      </div>
                    )}
                  </div>
                </div>
              </div>
            </div>
          )}

          {/* --- NEW CHAT VIEW --- */}
          {activeView === 'chat' && (
            <div className="flex-1 flex flex-col h-full bg-white relative">
              
              {/* Chat Header: Document Selector */}
              <div className="px-8 py-4 border-b border-slate-200 bg-slate-50 flex items-center justify-between shadow-sm z-10">
                <div>
                  <h3 className="font-bold text-slate-900">Legal Assistant</h3>
                  <p className="text-xs text-slate-500">
                    {sessionId ? "Continuing conversation" : "Select a document below or ask a general question"}
                  </p>
                </div>
                
                <div className="flex items-center gap-3">
                  <label className="text-sm font-medium text-slate-700">Target Document:</label>
                  <select 
                    className="w-64 bg-white border border-slate-300 rounded-lg px-3 py-2 text-sm focus:outline-none focus:border-blue-700 shadow-sm disabled:bg-slate-100 disabled:cursor-not-allowed"
                    value={selectedDocId || (documents.find(d => d.knowledge_base_id === selectedKbId)?.document_id || '')}
                    onChange={handleDocSelection}
                    disabled={!!sessionId} 
                  >
                    <option value="">-- General Chat (No Document) --</option>
                    {documents.map((doc, idx) => (
                      <option 
                        key={idx} 
                        value={doc.document_id}
                        disabled={doc.status === 'processing' || doc.status === 'pending' || doc.status === 'failed'}
                      >
                        {doc.title} {doc.status === 'processing' || doc.status === 'pending' ? '(Processing...)' : ''}
                      </option>
                    ))}
                  </select>
                </div>
              </div>

              {/* Chat Messages Area */}
              <div ref={chatScrollRef} className="flex-1 overflow-y-auto p-8 space-y-6 bg-white">
                {chatMessages.length === 0 ? (
                  <div className="h-full flex flex-col items-center justify-center text-slate-400">
                    <svg className="w-16 h-16 mb-4 text-blue-100" fill="currentColor" viewBox="0 0 24 24"><path d="M12 2C6.48 2 2 6.48 2 12s4.48 10 10 10 10-4.48 10-10S17.52 2 12 2zm0 18c-4.41 0-8-3.59-8-8s3.59-8 8-8 8 3.59 8 8-3.59 8-8 8zm-1-12h2v6h-2zm0 8h2v2h-2z"></path></svg>
                    <p className="text-lg font-medium text-slate-600">
                      {sessionId 
                        ? (isChatLoading ? "Loading past conversation..." : "Start your conversation.") 
                        : (selectedDocId ? "How can I help you with this document?" : "How can I help you today?")
                      }
                    </p>
                    <p className="text-sm mt-1">
                      {sessionId ? "" : "Select a document to analyze, or ask a general legal question."}
                    </p>
                  </div>
                ) : (
                  chatMessages.map((msg, index) => (
                    <div key={index} className={`flex ${msg.role === 'user' ? 'justify-end' : 'justify-start'}`}>
                      <div className={`max-w-[75%] px-5 py-3 rounded-2xl text-sm leading-relaxed shadow-sm whitespace-pre-wrap ${
                        msg.role === 'user' 
                          ? 'bg-blue-700 text-white rounded-br-none' 
                          : 'bg-slate-100 text-slate-800 border border-slate-200 rounded-bl-none'
                      }`}>
                        {msg.content}
                      </div>
                    </div>
                  ))
                )}
                
                {/* Typing Indicator */}
                {isChatLoading && (
                  <div className="flex justify-start">
                    <div className="bg-slate-100 border border-slate-200 px-5 py-4 rounded-2xl rounded-bl-none shadow-sm flex gap-1.5 items-center">
                      <div className="w-2 h-2 bg-slate-400 rounded-full animate-bounce" style={{ animationDelay: '0ms' }}></div>
                      <div className="w-2 h-2 bg-slate-400 rounded-full animate-bounce" style={{ animationDelay: '150ms' }}></div>
                      <div className="w-2 h-2 bg-slate-400 rounded-full animate-bounce" style={{ animationDelay: '300ms' }}></div>
                    </div>
                  </div>
                )}
              </div>

              {/* Chat Input Area */}
              <div className="p-6 bg-white border-t border-slate-200 shadow-[0_-4px_6px_-1px_rgba(0,0,0,0.05)]">
                <div className="max-w-4xl mx-auto relative flex items-center">
                  <input 
                    type="text"
                    value={chatInput}
                    onChange={(e) => setChatInput(e.target.value)}
                    onKeyDown={(e) => e.key === 'Enter' && handleSendMessage()}
                    disabled={isChatLoading}
                    placeholder={selectedDocId ? "Ask a question about the document..." : "Ask a general legal question..."}
                    className="w-full bg-slate-50 border border-slate-300 rounded-xl pl-5 pr-14 py-4 text-sm focus:outline-none focus:border-blue-700 focus:bg-white shadow-sm disabled:bg-slate-100 disabled:cursor-not-allowed"
                  />
                  <button 
                    onClick={handleSendMessage}
                    disabled={!chatInput.trim() || isChatLoading}
                    className="absolute right-3 p-2 bg-blue-700 hover:bg-blue-800 text-white rounded-lg transition-colors disabled:bg-slate-300 disabled:cursor-not-allowed"
                  >
                    <svg className="w-5 h-5" fill="none" stroke="currentColor" viewBox="0 0 24 24"><path strokeLinecap="round" strokeLinejoin="round" strokeWidth="2" d="M12 19l9 2-9-18-9 18 9-2zm0 0v-8"></path></svg>
                  </button>
                </div>
                <p className="text-center text-[11px] text-slate-400 mt-3">AI can make mistakes. Consider verifying important legal information.</p>
              </div>
            </div>
          )}

        </div>
      </main>

      {/* ================= UPLOAD MODAL (HIDDEN BY DEFAULT) ================= */}
      {showUploadModal && (
        <div className="fixed inset-0 z-50 flex items-center justify-center bg-slate-900/40 backdrop-blur-sm p-4">
          <div className="w-full max-w-lg bg-white rounded-3xl shadow-2xl border border-slate-100 overflow-hidden">
            
            <div className="px-6 py-5 border-b border-slate-100 flex justify-between items-center bg-slate-50/50">
              <h2 className="text-lg font-bold text-slate-900 tracking-tight">Upload Knowledge Base</h2>
              <button onClick={() => setShowUploadModal(false)} className="text-slate-400 hover:text-slate-700 bg-white hover:bg-slate-100 rounded-full p-1.5 transition-colors border border-transparent hover:border-slate-200">
                <svg className="w-5 h-5" fill="none" stroke="currentColor" viewBox="0 0 24 24"><path strokeLinecap="round" strokeLinejoin="round" strokeWidth="2" d="M6 18L18 6M6 6l12 12"></path></svg>
              </button>
            </div>

            <div className="p-6">
              <div 
                className={`relative border-2 border-dashed rounded-2xl p-8 text-center transition-all duration-200 cursor-pointer
                  ${isDragging ? 'border-blue-700 bg-blue-50' : 'border-slate-300 hover:border-blue-400 hover:bg-slate-50'}
                  ${file ? 'border-emerald-500 bg-emerald-50' : ''}
                `}
                onDragOver={handleDragOver}
                onDragLeave={handleDragLeave}
                onDrop={handleDrop}
                onClick={() => fileInputRef.current?.click()}
              >
                <input type="file" accept=".pdf" className="hidden" ref={fileInputRef} onChange={handleFileSelect} />
                
                <div className="flex flex-col items-center justify-center space-y-3">
                  {file ? (
                    <>
                      <div className="p-3 bg-emerald-100 text-emerald-600 rounded-full shadow-sm">
                        <svg className="w-8 h-8" fill="none" stroke="currentColor" viewBox="0 0 24 24"><path strokeLinecap="round" strokeLinejoin="round" strokeWidth="2" d="M9 12l2 2 4-4m6 2a9 9 0 11-18 0 9 9 0 0118 0z"></path></svg>
                      </div>
                      <div className="text-sm font-bold text-slate-900">{file.name}</div>
                      <div className="text-xs font-medium text-slate-500">{(file.size / 1024 / 1024).toFixed(2)} MB</div>
                    </>
                  ) : (
                    <>
                      <div className="p-3 bg-blue-50 text-blue-700 rounded-full shadow-sm border border-blue-100">
                        <svg className="w-8 h-8" fill="none" stroke="currentColor" viewBox="0 0 24 24"><path strokeLinecap="round" strokeLinejoin="round" strokeWidth="2" d="M7 16a4 4 0 01-.88-7.903A5 5 0 1115.9 6L16 6a5 5 0 011 9.9M15 13l-3-3m0 0l-3 3m3-3v12"></path></svg>
                      </div>
                      <p className="text-sm font-medium text-slate-600">
                        Drag and drop your PDF here, or <span className="text-blue-700 font-bold hover:underline">click to browse</span>
                      </p>
                      <p className="text-xs font-medium text-slate-400">Only .pdf files are supported</p>
                    </>
                  )}
                </div>
              </div>
            </div>

            <div className="px-6 py-5 bg-slate-50 border-t border-slate-100 flex justify-end gap-3">
              <button 
                onClick={() => setShowUploadModal(false)}
                className="px-5 py-2.5 rounded-lg text-sm font-semibold text-slate-600 hover:text-slate-900 hover:bg-slate-200 transition-colors"
              >
                Cancel
              </button>
              <button 
                onClick={handleUpload}
                disabled={!file || uploadState === 'uploading'}
                className={`px-6 py-2.5 rounded-lg text-sm font-bold text-white transition-all shadow-sm
                  ${!file ? 'bg-blue-300 cursor-not-allowed' : 'bg-blue-700 hover:bg-blue-800'}
                  ${uploadState === 'uploading' ? 'opacity-75 cursor-wait' : ''}
                `}
              >
                {uploadState === 'uploading' ? 'Uploading...' : 'Upload PDF'}
              </button>
            </div>
          </div>
        </div>
      )}

    </div>
  );
}