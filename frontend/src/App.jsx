import { useState } from "react";
import Analysis from "./Analysis";
import "./App.css";

function App() {
  const [showAnalysis, setShowAnalysis] = useState(false);

  // Show ECG Analysis page
  if (showAnalysis) {
    return <Analysis />;
  }

  return (
    <div className="app">

      {/* ================= NAVBAR ================= */}
      <nav className="navbar">
        <div className="nav-container">

          <div className="logo">
            <div className="logo-icon">
              <svg
                viewBox="0 0 40 40"
                fill="none"
                xmlns="http://www.w3.org/2000/svg"
              >
                <path
                  d="M5 21H11L15 10L21 30L26 15L29 21H35"
                  stroke="currentColor"
                  strokeWidth="2.5"
                  strokeLinecap="round"
                  strokeLinejoin="round"
                />
              </svg>
            </div>

            <span>Heart AI</span>
          </div>

          <div className="nav-links">
            <a href="#home">Home</a>
            <a href="#features">Features</a>
            <a href="#about">About</a>
          </div>

          <button
            className="nav-button"
            onClick={() => setShowAnalysis(true)}
          >
            Start Analysis
          </button>

        </div>
      </nav>


      {/* ================= HERO ================= */}
      <section className="hero" id="home">
        <div className="hero-container">

          <div className="hero-content">

            <div className="badge">
              <span className="badge-dot"></span>
              AI-Powered Cardiac Analysis
            </div>

            <h1>
              Understand your heart
              <span> through AI.</span>
            </h1>

            <p>
              Heart AI analyzes ECG signals using artificial intelligence
              to provide fast, intelligent insights into cardiac health.
            </p>

            <div className="hero-buttons">

              <button
                className="primary-button"
                onClick={() => setShowAnalysis(true)}
              >
                Start Analysis
                <span>→</span>
              </button>

              <a href="#features" className="secondary-button">
                Explore Features
              </a>

            </div>

            {/* STATS */}
            <div className="stats">

              <div className="stat">
                <strong>12</strong>
                <span>ECG Leads</span>
              </div>

              <div className="stat">
                <strong>AI</strong>
                <span>Analysis</span>
              </div>

              <div className="stat">
                <strong>10s</strong>
                <span>ECG Recording</span>
              </div>

            </div>

          </div>


          {/* ================= ECG VISUAL ================= */}
          <div className="hero-visual">

            <div className="ecg-card">

              <div className="ecg-header">
                <div>
                  <span className="ecg-label">LIVE ECG</span>
                  <span className="ecg-status">
                    <span></span>
                    Ready
                  </span>
                </div>

                <span className="ecg-bpm">72 BPM</span>
              </div>

              <div className="ecg-wave-container">

                <div className="ecg-grid"></div>

                <svg
                  className="ecg-wave"
                  viewBox="0 0 600 160"
                  preserveAspectRatio="none"
                >
                  <path
                    d="
                      M0 85
                      L70 85
                      L80 85
                      L90 85
                      L100 70
                      L110 85
                      L120 85
                      L130 85
                      L140 85
                      L150 85
                      L160 85
                      L170 85
                      L180 85
                      L190 85
                      L200 85
                      L210 85
                      L220 85
                      L230 85
                      L240 85
                      L250 85
                      L260 85
                      L270 85
                      L280 85
                      L290 85
                      L300 85
                      L310 85
                      L320 85
                      L330 85
                      L340 85
                      L350 85
                      L360 85
                      L370 85
                      L380 85
                      L390 85
                      L400 85
                      L410 85
                      L420 85
                      L430 85
                      L440 85
                      L450 85
                      L460 85
                      L470 85
                      L480 85
                      L490 85
                      L500 85
                      L510 85
                      L520 85
                      L530 85
                      L540 85
                      L550 85
                      L560 85
                      L570 85
                      L580 85
                      L590 85
                      L600 85
                    "
                  />
                </svg>

              </div>

              <div className="ecg-footer">
                <span>Lead II</span>
                <span>25 mm/s</span>
                <span>10 mm/mV</span>
              </div>

            </div>


            {/* FLOATING AI CARD */}
            <div className="floating-card">

              <div className="floating-icon">
                ✦
              </div>

              <div>
                <strong>AI Analysis</strong>
                <span>Ready to analyze</span>
              </div>

            </div>

          </div>

        </div>
      </section>


      {/* ================= FEATURES ================= */}
      <section className="features" id="features">

        <div className="section-container">

          <div className="section-heading">

            <span className="section-tag">
              FEATURES
            </span>

            <h2>
              Intelligent cardiac insights.
            </h2>

            <p>
              Designed to make ECG analysis simple, fast, and accessible.
            </p>

          </div>


          <div className="feature-grid">

            <div className="feature-card">

              <div className="feature-icon">
                ♥
              </div>

              <h3>ECG Analysis</h3>

              <p>
                Analyze 12-lead ECG signals using an AI-powered
                deep learning model.
              </p>

            </div>


            <div className="feature-card">

              <div className="feature-icon">
                ✦
              </div>

              <h3>AI Prediction</h3>

              <p>
                Identify patterns in ECG signals and generate
                intelligent cardiac predictions.
              </p>

            </div>


            <div className="feature-card">

              <div className="feature-icon">
                ⚡
              </div>

              <h3>Fast Results</h3>

              <p>
                Get analysis results within seconds through
                a simple and responsive interface.
              </p>

            </div>

          </div>

        </div>

      </section>


      {/* ================= ABOUT ================= */}
      <section className="about" id="about">

        <div className="section-container">

          <div className="about-content">

            <div>

              <span className="section-tag">
                ABOUT HEART AI
              </span>

              <h2>
                Technology designed around your heart.
              </h2>

            </div>

            <p>
              Heart AI combines ECG signal processing, machine learning,
              and modern web technology to create an accessible cardiac
              analysis platform for educational and research purposes.
            </p>

          </div>

        </div>

      </section>


      {/* ================= FOOTER ================= */}
      <footer className="footer">

        <div className="footer-container">

          <div className="footer-logo">
            <div className="logo-icon">
              <svg
                viewBox="0 0 40 40"
                fill="none"
                xmlns="http://www.w3.org/2000/svg"
              >
                <path
                  d="M5 21H11L15 10L21 30L26 15L29 21H35"
                  stroke="currentColor"
                  strokeWidth="2.5"
                  strokeLinecap="round"
                  strokeLinejoin="round"
                />
              </svg>
            </div>

            <span>Heart AI</span>
          </div>

          <p>
            AI-powered cardiac analysis for education and research.
          </p>

          <span className="copyright">
            © 2026 Heart AI
          </span>

        </div>

      </footer>

    </div>
  );
}

export default App;

