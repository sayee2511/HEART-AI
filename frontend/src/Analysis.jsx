
import { useState } from "react";
import "./Analysis.css";

function Analysis() {
  const [age, setAge] = useState("");
  const [sex, setSex] = useState("");
  const [height, setHeight] = useState("");
  const [weight, setWeight] = useState("");

  const [datFile, setDatFile] = useState(null);
  const [heaFile, setHeaFile] = useState(null);

  const [result, setResult] = useState(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState("");

  // ============================================================
  // FILE HANDLERS
  // ============================================================

  const handleDatFile = (event) => {
    const file = event.target.files[0];

    if (!file) return;

    setDatFile(file);
    setResult(null);
    setError("");
  };

  const handleHeaFile = (event) => {
    const file = event.target.files[0];

    if (!file) return;

    setHeaFile(file);
    setResult(null);
    setError("");
  };

  // ============================================================
  // ANALYZE ECG
  // ============================================================

  const handleAnalyze = async () => {
    setError("");
    setResult(null);

    if (!datFile || !heaFile) {
      setError(
        "Please upload both the .dat and .hea ECG files."
      );
      return;
    }

    const datName = datFile.name.replace(/\.dat$/i, "");
    const heaName = heaFile.name.replace(/\.hea$/i, "");

    if (datName !== heaName) {
      setError(
        "The .dat and .hea files must belong to the same ECG record."
      );
      return;
    }

    setLoading(true);

    try {
      const formData = new FormData();

      formData.append("dat_file", datFile);
      formData.append("hea_file", heaFile);

      const response = await fetch(
        "http://127.0.0.1:8000/predict",
        {
          method: "POST",
          body: formData,
        }
      );

      const data = await response.json();

      if (!response.ok) {
        throw new Error(
          data.detail ||
            `Server returned ${response.status}`
        );
      }

      setResult(data);
    } catch (err) {
      setError(
        err.message ||
          "Unable to connect to the Heart AI backend."
      );
    } finally {
      setLoading(false);
    }
  };

  // ============================================================
  // RENDER
  // ============================================================

  return (
    <div className="analysis-page">

      {/* ======================================================
          HEADER
      ====================================================== */}

      <header className="analysis-header">

        <div className="analysis-logo">

          <div className="analysis-logo-icon">
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

          <span>
            Heart <b>AI</b>
          </span>

        </div>

        <div className="system-status">
          <span></span>
          System Ready
        </div>

      </header>


      {/* ======================================================
          MAIN
      ====================================================== */}

      <main className="analysis-container">

        {/* ====================================================
            SIDEBAR
        ==================================================== */}

        <aside className="analysis-sidebar">

          <div className="sidebar-title">
            ECG ANALYSIS
          </div>


          <div className="step active">

            <div className="step-number">
              1
            </div>

            <div>
              <strong>
                Patient
              </strong>

              <span>
                Information
              </span>
            </div>

          </div>


          <div className="step-line"></div>


          <div
            className={`step ${
              datFile && heaFile ? "active" : ""
            }`}
          >

            <div className="step-number">
              2
            </div>

            <div>
              <strong>
                ECG
              </strong>

              <span>
                Upload
              </span>
            </div>

          </div>


          <div className="step-line"></div>


          <div
            className={`step ${
              result ? "active" : ""
            }`}
          >

            <div className="step-number">
              3
            </div>

            <div>
              <strong>
                AI
              </strong>

              <span>
                Analysis
              </span>
            </div>

          </div>

        </aside>


        {/* ====================================================
            CONTENT
        ==================================================== */}

        <section className="analysis-content">

          {/* ==================================================
              INTRO
          ================================================== */}

          <div className="analysis-intro">

            <div>

              <span className="analysis-label">
                ECG ANALYSIS
              </span>

              <h1>
                Patient <span>Information</span>
              </h1>

              <p>
                Enter basic information before ECG analysis.
              </p>

            </div>

            <div className="heart-decoration">
              ♡
            </div>

          </div>


          {/* ==================================================
              PATIENT INFORMATION
          ================================================== */}

          <section className="analysis-card">

            <div className="card-heading">

              <div className="card-icon">
                ♙
              </div>

              <div>

                <h2>
                  Patient Information
                </h2>

                <p>
                  Enter basic information before ECG analysis.
                </p>

              </div>

            </div>


            <div className="patient-grid">

              {/* AGE */}

              <div className="input-group">

                <label>
                  Age
                </label>

                <div className="input-wrapper">

                  <span>
                    ♙
                  </span>

                  <input
                    type="number"
                    placeholder="e.g. 45"
                    value={age}
                    onChange={(e) =>
                      setAge(e.target.value)
                    }
                  />

                </div>

              </div>


              {/* SEX */}

              <div className="input-group">

                <label>
                  Sex
                </label>

                <div className="input-wrapper">

                  <span>
                    ⚥
                  </span>

                  <select
                    value={sex}
                    onChange={(e) =>
                      setSex(e.target.value)
                    }
                  >

                    <option value="">
                      Select
                    </option>

                    <option value="Male">
                      Male
                    </option>

                    <option value="Female">
                      Female
                    </option>

                  </select>

                </div>

              </div>


              {/* HEIGHT */}

              <div className="input-group">

                <label>
                  Height (cm)
                </label>

                <div className="input-wrapper">

                  <span>
                    ↕
                  </span>

                  <input
                    type="number"
                    placeholder="e.g. 170"
                    value={height}
                    onChange={(e) =>
                      setHeight(e.target.value)
                    }
                  />

                </div>

              </div>


              {/* WEIGHT */}

              <div className="input-group">

                <label>
                  Weight (kg)
                </label>

                <div className="input-wrapper">

                  <span>
                    ♙
                  </span>

                  <input
                    type="number"
                    placeholder="e.g. 70"
                    value={weight}
                    onChange={(e) =>
                      setWeight(e.target.value)
                    }
                  />

                </div>

              </div>

            </div>

          </section>


          {/* ==================================================
              ECG UPLOAD
          ================================================== */}

          <section className="analysis-card">

            <div className="card-heading">

              <div className="card-icon">
                ↑
              </div>

              <div>

                <h2>
                  ECG Recording
                </h2>

                <p>
                  Upload the matching PTB-XL .dat and .hea files.
                </p>

              </div>

            </div>


            <div className="upload-grid">

              {/* DAT */}

              <label className="upload-box">

                <input
                  type="file"
                  accept=".dat"
                  onChange={handleDatFile}
                />

                <div className="upload-icon">
                  ↑
                </div>

                <strong>
                  {datFile
                    ? datFile.name
                    : "Upload .dat file"}
                </strong>

                <span>
                  {datFile
                    ? "ECG signal selected"
                    : "Click to browse"}
                </span>

              </label>


              {/* HEA */}

              <label className="upload-box">

                <input
                  type="file"
                  accept=".hea"
                  onChange={handleHeaFile}
                />

                <div className="upload-icon">
                  ↑
                </div>

                <strong>
                  {heaFile
                    ? heaFile.name
                    : "Upload .hea file"}
                </strong>

                <span>
                  {heaFile
                    ? "Header selected"
                    : "Click to browse"}
                </span>

              </label>

            </div>


            <div className="file-type">
              DAT <span>·</span> HEA
            </div>


            {/* SELECTED FILES */}

            <div className="selected-files">

              <div className="selected-heading">

                <div className="small-icon">
                  ▣
                </div>

                <div>

                  <strong>
                    Selected Files
                  </strong>

                  <span>
                    Your ECG files will appear here once uploaded.
                  </span>

                </div>

              </div>


              <div className="file-list">

                {datFile && (

                  <div className="file-item">

                    <span>
                      ▣
                    </span>

                    <div>

                      <strong>
                        {datFile.name}
                      </strong>

                      <small>
                        ECG signal
                      </small>

                    </div>

                  </div>

                )}


                {heaFile && (

                  <div className="file-item">

                    <span>
                      ▣
                    </span>

                    <div>

                      <strong>
                        {heaFile.name}
                      </strong>

                      <small>
                        ECG header
                      </small>

                    </div>

                  </div>

                )}


                {!datFile && !heaFile && (

                  <div className="empty-files">

                    <div>
                      ▱
                    </div>

                    <strong>
                      No files selected
                    </strong>

                    <span>
                      Please upload both .dat and .hea files
                    </span>

                  </div>

                )}

              </div>

            </div>

          </section>


          {/* ==================================================
              ECG PREVIEW
          ================================================== */}

          <section className="analysis-card preview-card">

            <div className="card-heading">

              <div className="card-icon">
                ◉
              </div>

              <div>

                <h2>
                  ECG Preview
                </h2>

                <p>
                  Uploaded ECG waveform visualization.
                </p>

              </div>

            </div>


            <div className="ecg-preview">

              <div className="ecg-line">

                <svg
                  viewBox="0 0 1000 180"
                  preserveAspectRatio="none"
                >

                  <polyline
                    points="
                      0,90
                      80,90
                      100,88
                      120,92
                      140,90
                      160,90
                      175,60
                      185,125
                      195,90
                      260,90
                      280,88
                      300,92
                      320,90
                      340,90
                      355,55
                      365,130
                      375,90
                      440,90
                      460,88
                      480,92
                      500,90
                      520,90
                      535,60
                      545,125
                      555,90
                      620,90
                      640,88
                      660,92
                      680,90
                      700,90
                      715,55
                      725,130
                      735,90
                      800,90
                      820,88
                      840,92
                      860,90
                      880,90
                      895,60
                      905,125
                      915,90
                      1000,90
                    "
                    fill="none"
                    stroke="currentColor"
                    strokeWidth="3"
                  />

                </svg>

              </div>

            </div>

          </section>


          {/* ==================================================
              ANALYZE BUTTON
          ================================================== */}

          <button
            className="analyze-button"
            onClick={handleAnalyze}
            disabled={
              loading ||
              !datFile ||
              !heaFile
            }
          >

            {loading
              ? "Analyzing ECG..."
              : "Analyze ECG →"}

          </button>


          {/* ==================================================
              ERROR
          ================================================== */}

          {error && (

            <div className="error-message">
              {error}
            </div>

          )}


          {/* ==================================================
              AI RESULTS
          ================================================== */}

          {result && (

            <section className="analysis-card result-card">

              {/* RESULT HEADER */}

              <div className="card-heading">

                <div className="card-icon">
                  ✓
                </div>

                <div>

                  <span className="analysis-label">
                    AI ANALYSIS
                  </span>

                  <h2>
                    ECG Analysis Complete
                  </h2>

                  <p>
                    Model prediction probabilities for the five
                    PTB-XL diagnostic superclass categories.
                  </p>

                </div>

              </div>


              {/* PRIMARY PREDICTION */}

              {result.predictions?.primary_prediction && (

                <div className="primary-result">

                  <div className="primary-result-label">
                    PRIMARY PREDICTION
                  </div>

                  <div className="primary-result-content">

                    <div>

                      <h3>
                        {
                          result.predictions
                            .primary_prediction
                            .name
                        }
                      </h3>

                      <p>
                        Model confidence:{" "}
                        {Number(
                          result.predictions
                            .primary_prediction
                            .percentage
                        ).toFixed(1)}
                        %
                      </p>

                    </div>

                    <div className="primary-confidence">

                      {Number(
                        result.predictions
                          .primary_prediction
                          .percentage
                      ).toFixed(1)}
                      %

                    </div>

                  </div>

                </div>

              )}


              {/* DETECTED CONDITIONS */}

              {result.predictions?.detected_conditions?.length > 0 && (

                <div className="detected-section">

                  <div className="section-mini-title">
                    DETECTED CATEGORIES
                  </div>

                  <div className="detected-list">

                    {result.predictions.detected_conditions.map(
                      (condition) => (

                        <span
                          className="detected-badge"
                          key={condition}
                        >
                          {condition}
                        </span>

                      )
                    )}

                  </div>

                </div>

              )}


              {result.predictions?.detected_conditions?.length === 0 && (

                <div className="no-detected">

                  No category crossed its configured
                  detection threshold.

                </div>

              )}


              {/* CATEGORY PROBABILITIES */}

              <div className="section-mini-title">
                CATEGORY PROBABILITIES
              </div>

              <div className="prediction-grid">

                {Object.entries(
                  result.predictions?.predictions || {}
                ).map(
                  ([label, prediction]) => (

                    <div
                      className="prediction-item"
                      key={label}
                    >

                      <div className="prediction-top">

                        <strong>
                          {prediction.name}
                        </strong>

                        <span>
                          {Number(
                            prediction.percentage
                          ).toFixed(1)}
                          %
                        </span>

                      </div>


                      <div className="progress-bar">

                        <div
                          style={{
                            width: `${Math.min(
                              Math.max(
                                Number(
                                  prediction.percentage
                                ) || 0,
                                0
                              ),
                              100
                            )}%`,
                          }}
                        />

                      </div>


                      <div className="prediction-meta">

                        <span>
                          Code: {label}
                        </span>

                        <span>
                          Threshold:{" "}
                          {Number(
                            prediction.threshold
                          ).toFixed(2)}
                        </span>

                      </div>

                    </div>

                  )
                )}

              </div>


              {/* DISCLAIMER */}

              <p className="result-note">

                These results are model predictions based on the
                PTB-XL diagnostic superclass categories. They are
                for educational and research purposes and are not
                a medical diagnosis.

              </p>

            </section>

          )}

        </section>

      </main>

    </div>
  );
}

export default Analysis;



                     