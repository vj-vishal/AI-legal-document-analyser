import React from 'react';
import { Link } from 'react-router-dom';

export default function LandingPage() {
  return (
    <div className="min-h-screen bg-slate-50 font-sans text-slate-900 selection:bg-blue-200">
      
      {/* Navigation Bar */}
      <nav className="sticky top-0 z-50 flex items-center justify-between px-6 py-4 bg-white/80 backdrop-blur-md border-b border-slate-200">
        <div className="flex items-center space-x-2">
          <div className="w-8 h-8 bg-blue-700 text-white flex items-center justify-center rounded-lg font-bold text-xl">
            ⚖️
          </div>
          <span className="text-2xl font-bold text-slate-900 tracking-tight">ApnaKanoon</span>
        </div>
        <div className="hidden md:flex space-x-8 text-sm font-medium text-slate-600">
          <a href="#features" className="hover:text-blue-700 transition-colors">Features</a>
          <a href="#how-it-works" className="hover:text-blue-700 transition-colors">How it Works</a>
          {/* <a href="#pricing" className="hover:text-blue-700 transition-colors">Pricing</a> */}
        </div>
        <div className="flex items-center space-x-4 text-sm font-medium">
          {/* THESE ARE THE UPDATED LINKS */}
          <Link to="/auth" state={{ mode: 'login' }} className="text-slate-700 hover:text-blue-700 transition-colors">
            Log In
          </Link>
          <Link to="/auth" state={{ mode: 'signup' }} className="px-5 py-2.5 bg-blue-700 text-white rounded-lg hover:bg-blue-800 transition-all shadow-sm">
            Sign Up Free
          </Link>
        </div>
      </nav>

      {/* Hero Section */}
      <header className="relative max-w-6xl mx-auto px-6 py-24 md:py-32 flex flex-col items-center text-center">
        <div className="inline-flex items-center space-x-2 px-3 py-1 bg-blue-50 text-blue-700 rounded-full text-sm font-semibold mb-8 border border-blue-100">
          <span className="flex h-2 w-2 rounded-full bg-blue-600"></span>
          <span>Trained on Indian Law (IPC, BNS, CrPC)</span>
        </div>
        
        <h1 className="text-5xl md:text-6xl font-extrabold tracking-tight text-slate-900 mb-6 leading-tight">
          Legal Intelligence <br className="hidden md:block"/> Made Simple.
        </h1>
        
        <p className="text-xl text-slate-600 mb-10 max-w-2xl mx-auto leading-relaxed">
          Ask anything about Indian law. Get answers grounded in statutes, judgments, and legal procedures in any Indian language. Built for citizens, lawyers, and law students.
        </p>

        <div className="w-full max-w-2xl bg-white p-2 rounded-2xl shadow-xl border border-slate-200 flex items-center">
          <input 
            type="text" 
            placeholder="E.g. My landlord hasn't returned my security deposit..." 
            className="flex-1 bg-transparent px-4 py-3 text-slate-800 placeholder-slate-400 focus:outline-none text-lg"
          />
          <button className="px-6 py-3 bg-slate-900 text-white rounded-xl font-semibold hover:bg-slate-800 transition-colors flex items-center space-x-2">
            <span>Ask AI</span>
            <span>→</span>
          </button>
        </div>
        <p className="mt-4 text-sm text-slate-500">100 free queries every month. No credit card required.</p>
      </header>

      {/* Features Grid */}
      <section id="features" className="bg-white py-24 border-t border-slate-200">
        <div className="max-w-6xl mx-auto px-6">
          <div className="text-center mb-16">
            <h2 className="text-3xl md:text-4xl font-bold text-slate-900 mb-4">Everything You Need for Legal Clarity</h2>
            <p className="text-lg text-slate-600">Enterprise-grade tools simplified for everyday legal research.</p>
          </div>

          <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-8">
            <FeatureCard icon="💬" title="AI Legal Chat" desc="Ask in Hindi, get answers in English. Multi-language support with voice input for natural conversations." />
            <FeatureCard icon="📄" title="Document Intelligence" desc="Upload contracts, notices, or case files (PDF/DOCX). Ask questions and get AI-powered OCR analysis instantly." />
            <FeatureCard icon="⚖️" title="Case Law Grounding" desc="Answers backed by the Constitution and thousands of Supreme Court and High Court judgments." />
            <FeatureCard icon="📁" title="Case Management" desc="Keep your legal research organized by case or client in private collections. Find any insight in seconds." />
            <FeatureCard icon="🔒" title="100% Private" desc="Enterprise-grade security. Your chats and documents are end-to-end encrypted and not used for AI training." />
            <FeatureCard icon="🌐" title="Live Web Search" desc="Search the latest judgments, SC/HC orders, notifications, and legal news with verifiable sources." />
          </div>
        </div>
      </section>

      {/* How It Works Section */}
      <section id="how-it-works" className="bg-slate-900 text-white py-24">
        <div className="max-w-6xl mx-auto px-6">
          <div className="text-center mb-16">
            <h2 className="text-3xl md:text-4xl font-bold mb-4">Legal Help in 3 Simple Steps</h2>
            <p className="text-slate-400 text-lg">No appointments, no waiting rooms, no confusing legal jargon.</p>
          </div>

          <div className="grid grid-cols-1 md:grid-cols-3 gap-12">
            <StepCard number="01" title="Ask Your Question" desc="Describe your situation the way you'd explain it to a friend. The AI translates it into legal context automatically." />
            <StepCard number="02" title="Get AI Analysis" desc="We cross-reference your query against Indian legal codes (IPC, BNS, CrPC) and relevant court precedents." />
            <StepCard number="03" title="Receive Guidance" desc="Get specific sections of applicable laws, step-by-step next actions, and template notices if required." />
          </div>
        </div>
      </section>

      <footer className="bg-white py-12 border-t border-slate-200 text-center">
        <div className="flex items-center justify-center space-x-2 mb-4">
          <div className="w-6 h-6 bg-blue-700 text-white flex items-center justify-center rounded font-bold text-xs">⚖️</div>
          <span className="text-xl font-bold text-slate-900">ApnaKanoon</span>
        </div>
        <p className="text-slate-500 text-sm">© 2026 Legal AI Platform. Built for Indian Law.</p>
      </footer>
    </div>
  );
}

function FeatureCard({ icon, title, desc }) {
  return (
    <div className="p-8 rounded-2xl bg-slate-50 border border-slate-100 hover:border-blue-200 hover:shadow-lg hover:-translate-y-1 transition-all duration-300">
      <div className="text-4xl mb-6">{icon}</div>
      <h3 className="text-xl font-bold text-slate-900 mb-3">{title}</h3>
      <p className="text-slate-600 leading-relaxed">{desc}</p>
    </div>
  );
}

function StepCard({ number, title, desc }) {
  return (
    <div className="relative p-6">
      <div className="text-6xl font-black text-slate-800 absolute -top-6 left-6 opacity-50">{number}</div>
      <div className="relative z-10 pt-8">
        <h3 className="text-xl font-bold text-white mb-4">{title}</h3>
        <p className="text-slate-400 leading-relaxed">{desc}</p>
      </div>
    </div>
  );
}