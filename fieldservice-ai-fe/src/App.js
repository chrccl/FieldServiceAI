import React, { useState, useRef, useCallback } from 'react';
import { 
  startInspection, 
  uploadFile, 
  generateReport,
  getReportStatus,
  downloadReport
} from './services/api';
import { 
  Camera, 
  Mic, 
  Upload, 
  FileText, 
  AlertTriangle, 
  CheckCircle, 
  Download,
  Play,
  Pause,
  Square,
  Eye,
  Zap,
  Shield,
  Clock,
  Users,
  TrendingUp
} from 'lucide-react';

const FieldServiceAI = () => {
  const [activeTab, setActiveTab] = useState('capture');
  const [uploadedFiles, setUploadedFiles] = useState([]);
  const [voiceNote, setVoiceNote] = useState('');
  const [isRecording, setIsRecording] = useState(false);
  const [analysisResults, setAnalysisResults] = useState(null);
  const [reportGenerated, setReportGenerated] = useState(false);
  const [processing, setProcessing] = useState(false);
  const fileInputRef = useRef(null);
  const [recordingTime, setRecordingTime] = useState(0);
  const recordingInterval = useRef(null);
  const [reportId, setReportId] = useState(null);
  const [processingJobs, setProcessingJobs] = useState([]);
  const [notifications, setNotifications] = useState([]);

  // Initialize inspection on first data capture
  useEffect(() => {
    if ((uploadedFiles.length > 0 || voiceNote) && !reportId) {
      initInspection();
    }
  }, [uploadedFiles, voiceNote]);

  const initInspection = async () => {
    const inspectionData = {
      site_name: "Industrial Facility Block A",
      technician_id: "tech-12345",
      technician_name: "John Fieldworker",
      inspection_type: "routine"
    };
    
    const response = await startInspection(inspectionData);
    setReportId(response.report_id);
    setProcessingJobs([...processingJobs, { id: response.job_id, type: 'inspection_init' }]);
  };

  const showNotification = (message, type = 'info') => {
    const id = Date.now();
    setNotifications(prev => [...prev, { id, message, type }]);
    setTimeout(() => {
      setNotifications(prev => prev.filter(n => n.id !== id));
    }, 5000);
  };

  const handleFileUpload = async (event) => {
    const files = Array.from(event.target.files);
    const newFiles = files.map(file => ({
      id: Date.now() + Math.random(),
      name: file.name,
      type: file.type,
      size: file.size,
      url: URL.createObjectURL(file),
      file // store actual file object
    }));
    
    setUploadedFiles(prev => [...prev, ...newFiles]);
    
    // Upload to backend
    if (reportId) {
      newFiles.forEach(async (file) => {
        let fileType = 'document';
        if (file.type.startsWith('image/') || file.type.startsWith('video/')) {
          fileType = 'image';
        }
        
        const response = await uploadFile(reportId, fileType, file.file);
        setProcessingJobs(prev => [...prev, {
          id: response.data.job_id,
          type: `${fileType}_analysis`,
          status: 'queued'
        }]);
      });
    }
  };

  const startRecording = () => {
    setIsRecording(true);
    setRecordingTime(0);
    recordingInterval.current = setInterval(() => {
      setRecordingTime(prev => prev + 1);
    }, 1000);
  };

  const stopRecording = () => {
    setIsRecording(false);
    clearInterval(recordingInterval.current);
    setVoiceNote("Recording completed - analyzing audio...");

    if (mediaRecorderRef.current && reportId) {
      mediaRecorderRef.current.stop();
      const audioBlob = new Blob(audioChunksRef.current, { type: 'audio/wav' });
      const audioFile = new File([audioBlob], 'voice-note.wav');
      
      uploadFile(reportId, 'audio', audioFile).then(response => {
        setProcessingJobs(prev => [...prev, {
          id: response.data.job_id,
          type: 'audio_analysis',
          status: 'queued'
        }]);
      });
    }
  };

  const runAnalysis = async () => {
    setProcessing(true);
    setActiveTab('analysis');
    
    // Start report generation
    const response = await generateReport(reportId);
    setProcessingJobs(prev => [...prev, {
      id: response.data.job_id,
      type: 'report_generation',
      status: 'queued'
    }]);
  
    // Start polling for results
    pollReportStatus();
  };

  const pollReportStatus = async () => {
    const interval = setInterval(async () => {
      const response = await getReportStatus(reportId);
      const reportData = response.data;
      
      if (reportData.status === 'completed') {
        clearInterval(interval);
        setProcessing(false);
        setAnalysisResults(JSON.parse(reportData.analysis_results));
      }
    }, 3000);
  };

  const generateReport = async () => {
    setReportGenerated(true);
    setActiveTab('report');
    
    // Trigger report generation if not already done
    if (!processingJobs.some(job => job.type === 'report_generation')) {
      const response = await generateReport(reportId);
      setProcessingJobs(prev => [...prev, {
        id: response.data.job_id,
        type: 'report_generation',
        status: 'queued'
      }]);
    }
  };

  const formatTime = (seconds) => {
    const mins = Math.floor(seconds / 60);
    const secs = seconds % 60;
    return `${mins}:${secs.toString().padStart(2, '0')}`;
  };

  const getSeverityColor = (severity) => {
    switch(severity) {
      case 'Critical': return 'text-red-600 bg-red-50';
      case 'High': return 'text-orange-600 bg-orange-50';
      case 'Medium': return 'text-yellow-600 bg-yellow-50';
      default: return 'text-green-600 bg-green-50';
    }
  };

  return (
    <div className="min-h-screen bg-gradient-to-br from-blue-50 via-white to-indigo-50">
      {/* Header */}
      <div className="bg-white shadow-lg border-b border-gray-200">
        <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8">
          <div className="flex items-center justify-between h-16">
            <div className="flex items-center space-x-3">
              <div className="w-10 h-10 bg-gradient-to-r from-blue-600 to-indigo-600 rounded-xl flex items-center justify-center">
                <Zap className="w-6 h-6 text-white" />
              </div>
              <div>
                <h1 className="text-xl font-bold bg-gradient-to-r from-blue-600 to-indigo-600 bg-clip-text text-transparent">
                  FieldServiceAI
                </h1>
                <p className="text-sm text-gray-500">Intelligent Field Inspection Platform</p>
              </div>
            </div>
            <div className="flex items-center space-x-4">
              <div className="flex items-center space-x-2 text-sm text-gray-600">
                <Shield className="w-4 h-4" />
                <span>SOC2 Compliant</span>
              </div>
              <div className="w-8 h-8 bg-gray-300 rounded-full"></div>
            </div>
          </div>
        </div>
      </div>

      {/* Stats Dashboard */}
      <div className="grid grid-cols-2 md:grid-cols-4 gap-4 mb-8">
        <div className="grid grid-cols-1 md:grid-cols-4 gap-6 mb-8">
          <div className="bg-white rounded-xl shadow-sm p-6 border border-gray-100">
            <div className="flex items-center justify-between">
              <div>
                <p className="text-sm text-gray-600">Time Saved</p>
                <p className="text-2xl font-bold text-green-600">85%</p>
              </div>
              <Clock className="w-8 h-8 text-green-500" />
            </div>
          </div>
          <div className="bg-white rounded-xl shadow-sm p-6 border border-gray-100">
            <div className="flex items-center justify-between">
              <div>
                <p className="text-sm text-gray-600">Error Reduction</p>
                <p className="text-2xl font-bold text-blue-600">92%</p>
              </div>
              <TrendingUp className="w-8 h-8 text-blue-500" />
            </div>
          </div>
          <div className="bg-white rounded-xl shadow-sm p-6 border border-gray-100">
            <div className="flex items-center justify-between">
              <div>
                <p className="text-sm text-gray-600">Daily Inspections</p>
                <p className="text-2xl font-bold text-purple-600">127</p>
              </div>
              <Eye className="w-8 h-8 text-purple-500" />
            </div>
          </div>
          <div className="bg-white rounded-xl shadow-sm p-6 border border-gray-100">
            <div className="flex items-center justify-between">
              <div>
                <p className="text-sm text-gray-600">Active Users</p>
                <p className="text-2xl font-bold text-indigo-600">234</p>
              </div>
              <Users className="w-8 h-8 text-indigo-500" />
            </div>
          </div>
        </div>

        {/* Navigation Tabs */}
        <div className="flex flex-col sm:flex-row border-b border-gray-200">
          <div className="flex border-b border-gray-200">
            {[
              { id: 'capture', label: 'Data Capture', icon: Camera },
              { id: 'analysis', label: 'AI Analysis', icon: Zap },
              { id: 'report', label: 'Report Generation', icon: FileText }
            ].map(tab => {
              const Icon = tab.icon;
              return (
                <button
                  key={tab.id}
                  onClick={() => setActiveTab(tab.id)}
                  className={`flex-1 flex items-center justify-center space-x-2 py-4 px-6 text-sm font-medium transition-colors ${
                    activeTab === tab.id
                      ? 'text-blue-600 border-b-2 border-blue-600 bg-blue-50'
                      : 'text-gray-600 hover:text-gray-900 hover:bg-gray-50'
                  }`}
                >
                  <Icon className="w-5 h-5" />
                  <span>{tab.label}</span>
                </button>
              );
            })}
          </div>

          <div className="p-8">
            {/* Data Capture Tab */}
            {activeTab === 'capture' && (
              <div className="space-y-8">
                <div>
                  <h2 className="text-2xl font-bold text-gray-900 mb-2">Field Data Capture</h2>
                  <p className="text-gray-600">Upload photos, videos, PDFs and record voice notes for comprehensive inspection documentation.</p>
                </div>

                <div className="grid grid-cols-1 lg:grid-cols-2 gap-8">
                  {/* File Upload Section */}
                  <div className="space-y-6">
                    <h3 className="text-lg font-semibold text-gray-900 flex items-center space-x-2">
                      <Upload className="w-5 h-5" />
                      <span>Media & Documents</span>
                    </h3>
                    
                    <div 
                      onClick={() => fileInputRef.current?.click()}
                      className="border-2 border-dashed border-gray-300 rounded-xl p-8 text-center hover:border-blue-400 hover:bg-blue-50 transition-colors cursor-pointer"
                    >
                      <Camera className="w-12 h-12 text-gray-400 mx-auto mb-4" />
                      <p className="text-gray-600 mb-2">Drop files here or click to upload</p>
                      <p className="text-sm text-gray-400">Images, Videos, PDFs, Spec Sheets</p>
                    </div>
                    
                    <input
                      ref={fileInputRef}
                      type="file"
                      multiple
                      accept="image/*,video/*,.pdf"
                      onChange={handleFileUpload}
                      className="hidden"
                    />

                    {(uploadedFiles.length > 0 || voiceNote) && !reportId && (
                      <div className="space-y-3">
                        <h4 className="font-medium text-gray-900">Uploaded Files</h4>
                        <div className="space-y-2">
                          {uploadedFiles.map(file => (
                            <div key={file.id} className="flex items-center space-x-3 p-3 bg-gray-50 rounded-lg">
                              <FileText className="w-5 h-5 text-gray-400" />
                              <div className="flex-1">
                                <p className="text-sm font-medium text-gray-900">{file.name}</p>
                                <p className="text-xs text-gray-500">{(file.size / 1024 / 1024).toFixed(2)} MB</p>
                              </div>
                              <CheckCircle className="w-5 h-5 text-green-500" />
                            </div>
                          ))}
                        </div>
                        <div className="text-center py-4 text-gray-500">
                          <div className="w-8 h-8 border-4 border-blue-600 border-t-transparent rounded-full animate-spin mx-auto mb-2"></div>
                          <p>Initializing inspection...</p>
                        </div>
                      </div>
                    )}
                  </div>

                  {/* Voice Recording Section */}
                  <div className="space-y-6">
                    <h3 className="text-lg font-semibold text-gray-900 flex items-center space-x-2">
                      <Mic className="w-5 h-5" />
                      <span>Voice Notes</span>
                    </h3>
                    
                    <div className="bg-gray-50 rounded-xl p-6">
                      <div className="text-center mb-6">
                        {!isRecording ? (
                          <button
                            onClick={startRecording}
                            className="w-20 h-20 bg-red-500 hover:bg-red-600 rounded-full flex items-center justify-center transition-colors group"
                          >
                            <Mic className="w-8 h-8 text-white group-hover:scale-110 transition-transform" />
                          </button>
                        ) : (
                          <div className="space-y-4">
                            <div className="w-20 h-20 bg-red-500 rounded-full flex items-center justify-center animate-pulse">
                              <Square className="w-8 h-8 text-white" />
                            </div>
                            <div className="text-red-600 font-mono text-lg">{formatTime(recordingTime)}</div>
                            <button
                              onClick={stopRecording}
                              className="px-4 py-2 bg-gray-800 text-white rounded-lg hover:bg-gray-900 transition-colors"
                            >
                              Stop Recording
                            </button>
                          </div>
                        )}
                      </div>
                      
                      {voiceNote && (
                        <div className="bg-white rounded-lg p-4 border border-gray-200">
                          <h4 className="font-medium text-gray-900 mb-2">Transcription</h4>
                          <p className="text-gray-700 text-sm">{voiceNote}</p>
                        </div>
                      )}
                    </div>
                  </div>
                </div>

                {(uploadedFiles.length > 0 || voiceNote) && (
                  <div className="flex justify-center pt-6">
                    <button
                      onClick={runAnalysis}
                      className="px-8 py-3 bg-gradient-to-r from-blue-600 to-indigo-600 text-white rounded-xl hover:from-blue-700 hover:to-indigo-700 transition-all transform hover:scale-105 flex items-center space-x-2"
                    >
                      <Zap className="w-5 h-5" />
                      <span>Run AI Analysis</span>
                    </button>
                  </div>
                )}
              </div>
            )}

            {/* Analysis Tab */}
            {processing && (
              <div className="text-center py-12">
                <div className="w-16 h-16 border-4 border-blue-600 border-t-transparent rounded-full animate-spin mx-auto mb-4"></div>
                <p className="text-gray-600">Processing multi-modal data with AI models...</p>
                <div className="mt-4 space-y-2 text-sm text-gray-500">
                  {processingJobs.filter(job => job.status !== 'completed').map(job => (
                    <p key={job.id}>⏳ {job.type.replace('_', ' ')} in progress</p>
                  ))}
                </div>
              </div>
            )}
            {activeTab === 'analysis' && (
              <div className="space-y-8">
                <div>
                  <h2 className="text-2xl font-bold text-gray-900 mb-2">AI-Powered Analysis</h2>
                  <p className="text-gray-600">Advanced computer vision, speech recognition, and document analysis results.</p>
                </div>

                {processing ? (
                  <div className="text-center py-12">
                    <div className="w-16 h-16 border-4 border-blue-600 border-t-transparent rounded-full animate-spin mx-auto mb-4"></div>
                    <p className="text-gray-600">Processing multi-modal data with AI models...</p>
                    <div className="mt-4 space-y-2 text-sm text-gray-500">
                      <p>🔍 Object detection & segmentation</p>
                      <p>🎤 Speech recognition & classification</p>
                      <p>📄 Document parsing & extraction</p>
                    </div>
                  </div>
                ) : analysisResults ? (
                  <div className="space-y-8">
                    {/* Object Detection Results */}
                    <div className="bg-white rounded-xl border border-gray-200 p-6">
                      <h3 className="text-lg font-semibold text-gray-900 mb-4 flex items-center space-x-2">
                        <Eye className="w-5 h-5 text-blue-600" />
                        <span>Computer Vision Detection</span>
                      </h3>
                      <div className="space-y-3">
                        {analysisResults.objectDetection.map((detection, idx) => (
                          <div key={idx} className={`flex items-center justify-between p-4 rounded-lg border ${getSeverityColor(detection.severity)}`}>
                            <div className="flex items-center space-x-3">
                              <AlertTriangle className="w-5 h-5" />
                              <div>
                                <p className="font-medium">{detection.type}</p>
                                <p className="text-sm opacity-75">{detection.location}</p>
                              </div>
                            </div>
                            <div className="text-right">
                              <p className="font-semibold">{detection.severity}</p>
                              <p className="text-sm opacity-75">{(detection.confidence * 100).toFixed(1)}% confident</p>
                            </div>
                          </div>
                        ))}
                      </div>
                    </div>

                    {/* Voice Analysis Results */}
                    <div className="bg-white rounded-xl border border-gray-200 p-6">
                      <h3 className="text-lg font-semibold text-gray-900 mb-4 flex items-center space-x-2">
                        <Mic className="w-5 h-5 text-green-600" />
                        <span>Voice Analysis & NLP</span>
                      </h3>
                      <div className="space-y-4">
                        <div>
                          <h4 className="font-medium text-gray-900 mb-2">Transcript</h4>
                          <p className="text-gray-700 bg-gray-50 p-3 rounded-lg text-sm">{analysisResults.voiceAnalysis.transcript}</p>
                        </div>
                        <div>
                          <h4 className="font-medium text-gray-900 mb-2">Key Insights</h4>
                          <ul className="space-y-2">
                            {analysisResults.voiceAnalysis.keyInsights.map((insight, idx) => (
                              <li key={idx} className="flex items-start space-x-2">
                                <CheckCircle className="w-4 h-4 text-green-500 mt-0.5 flex-shrink-0" />
                                <span className="text-sm text-gray-700">{insight}</span>
                              </li>
                            ))}
                          </ul>
                        </div>
                      </div>
                    </div>

                    {/* Document Analysis */}
                    <div className="bg-white rounded-xl border border-gray-200 p-6">
                      <h3 className="text-lg font-semibold text-gray-900 mb-4 flex items-center space-x-2">
                        <FileText className="w-5 h-5 text-purple-600" />
                        <span>Document & Spec Analysis</span>
                      </h3>
                      <div className="grid md:grid-cols-2 gap-6">
                        <div>
                          <h4 className="font-medium text-gray-900 mb-3">Maintenance Thresholds</h4>
                          <div className="space-y-2">
                            {Object.entries(analysisResults.documentExtraction.maintenanceThresholds).map(([key, value]) => (
                              <div key={key} className="flex justify-between text-sm">
                                <span className="text-gray-600">{key}:</span>
                                <span className="font-medium text-gray-900">{value}</span>
                              </div>
                            ))}
                          </div>
                        </div>
                        <div>
                          <h4 className="font-medium text-gray-900 mb-3">Compliance Status</h4>
                          <div className="text-sm text-red-600 bg-red-50 p-3 rounded-lg border border-red-200">
                            {analysisResults.documentExtraction.complianceStatus}
                          </div>
                        </div>
                      </div>
                    </div>

                    <div className="flex justify-center pt-6">
                      <button
                        onClick={generateReport}
                        className="px-8 py-3 bg-gradient-to-r from-green-600 to-emerald-600 text-white rounded-xl hover:from-green-700 hover:to-emerald-700 transition-all transform hover:scale-105 flex items-center space-x-2"
                      >
                        <FileText className="w-5 h-5" />
                        <span>Generate Report</span>
                      </button>
                    </div>
                  </div>
                ) : (
                  <div className="text-center py-12 text-gray-500">
                    <Zap className="w-16 h-16 mx-auto mb-4 opacity-30" />
                    <p>No analysis data available. Please capture data first.</p>
                  </div>
                )}
              </div>
            )}

            {/* Report Tab */}
            {activeTab === 'report' && (
              <div className="space-y-8">
                <div>
                  <h2 className="text-2xl font-bold text-gray-900 mb-2">Automated Report Generation</h2>
                  <p className="text-gray-600">Professional inspection reports generated from your field data and AI analysis.</p>
                </div>

                {reportGenerated ? (
                  <div className="space-y-8">
                    {/* Report Preview */}
                    <div className="bg-white rounded-xl border border-gray-200 p-8 shadow-sm">
                      <div className="border-b border-gray-200 pb-6 mb-6">
                        <div className="flex items-center justify-between">
                          <div>
                            <h3 className="text-2xl font-bold text-gray-900">Field Inspection Report</h3>
                            <p className="text-gray-600 mt-1">Site: Industrial Facility Block A | Date: {new Date().toLocaleDateString()}</p>
                          </div>
                          <div className="text-right">
                            <p className="text-sm text-gray-500">Report ID: FSA-{Date.now().toString().slice(-6)}</p>
                            <p className="text-sm text-gray-500">Generated by FieldServiceAI</p>
                          </div>
                        </div>
                      </div>

                      {/* Executive Summary */}
                      <div className="mb-8">
                        <h4 className="text-lg font-semibold text-gray-900 mb-3">Executive Summary</h4>
                        <div className="bg-red-50 border border-red-200 rounded-lg p-4 mb-4">
                          <div className="flex items-start space-x-3">
                            <AlertTriangle className="w-5 h-5 text-red-600 mt-0.5" />
                            <div>
                              <p className="font-medium text-red-800">Critical Issues Identified</p>
                              <p className="text-red-700 text-sm mt-1">
                                3 maintenance issues requiring immediate attention. Compliance status: Non-compliant with 2/3 safety parameters exceeded.
                              </p>
                            </div>
                          </div>
                        </div>
                        <p className="text-gray-700 text-sm leading-relaxed">
                          Comprehensive inspection revealed critical infrastructure concerns including valve leakage, pipe corrosion, and hardware maintenance requirements. 
                          Automated analysis indicates immediate intervention needed to prevent safety hazards and ensure operational continuity.
                        </p>
                      </div>

                      {/* Findings Checklist */}
                      <div className="mb-8">
                        <h4 className="text-lg font-semibold text-gray-900 mb-3">Detailed Findings</h4>
                        <div className="space-y-3">
                          <div className="flex items-start space-x-3 p-3 border border-red-200 rounded-lg bg-red-50">
                            <AlertTriangle className="w-5 h-5 text-red-600 mt-0.5" />
                            <div className="flex-1">
                              <p className="font-medium text-red-800">Critical: Valve C1 Leakage</p>
                              <p className="text-red-700 text-sm">Detected with 93% confidence. Requires immediate repair.</p>
                            </div>
                            <span className="text-xs bg-red-600 text-white px-2 py-1 rounded">URGENT</span>
                          </div>
                          <div className="flex items-start space-x-3 p-3 border border-orange-200 rounded-lg bg-orange-50">
                            <AlertTriangle className="w-5 h-5 text-orange-600 mt-0.5" />
                            <div className="flex-1">
                              <p className="font-medium text-orange-800">High: Pipe Joint A3 Corrosion</p>
                              <p className="text-orange-700 text-sm">89% confidence. Schedule preventive maintenance within 48 hours.</p>
                            </div>
                            <span className="text-xs bg-orange-600 text-white px-2 py-1 rounded">HIGH</span>
                          </div>
                          <div className="flex items-start space-x-3 p-3 border border-yellow-200 rounded-lg bg-yellow-50">
                            <AlertTriangle className="w-5 h-5 text-yellow-600 mt-0.5" />
                            <div className="flex-1">
                              <p className="font-medium text-yellow-800">Medium: Flange B2 Loose Bolt</p>
                              <p className="text-yellow-700 text-sm">76% confidence. Tighten to specification within 24 hours.</p>
                            </div>
                            <span className="text-xs bg-yellow-600 text-white px-2 py-1 rounded">MED</span>
                          </div>
                        </div>
                      </div>

                      {/* Recommendations */}
                      <div className="mb-8">
                        <h4 className="text-lg font-semibold text-gray-900 mb-3">Recommendations</h4>
                        <div className="bg-blue-50 border border-blue-200 rounded-lg p-4">
                          <ul className="space-y-2 text-sm text-blue-800">
                            <li className="flex items-start space-x-2">
                              <span className="font-medium text-blue-900">1.</span>
                              <span>Immediately isolate and repair Valve C1 to prevent safety hazards</span>
                            </li>
                            <li className="flex items-start space-x-2">
                              <span className="font-medium text-blue-900">2.</span>
                              <span>Schedule corrosion treatment for Pipe Joint A3 within 48 hours</span>
                            </li>
                            <li className="flex items-start space-x-2">
                              <span className="font-medium text-blue-900">3.</span>
                              <span>Tighten Flange B2 bolt to 85-95 Nm specification</span>
                            </li>
                            <li className="flex items-start space-x-2">
                              <span className="font-medium text-blue-900">4.</span>
                              <span>Implement weekly visual inspections for early detection</span>
                            </li>
                          </ul>
                        </div>
                      </div>

                      {/* Report Footer */}
                      <div className="border-t border-gray-200 pt-6 text-center text-sm text-gray-500">
                        <p>This report was automatically generated by FieldServiceAI using advanced computer vision, NLP, and document analysis.</p>
                        <p className="mt-1">For questions or technical support, contact: support@fieldserviceai.com</p>
                      </div>
                    </div>

                    {/* Action Buttons */}
                    <div className="flex justify-center space-x-4">
                      <a 
                        href={downloadReport(reportId, 'pdf')}
                        download={`FieldServiceAI_Report_${reportId}.pdf`}
                        className="px-6 py-3 bg-blue-600 text-white rounded-xl hover:bg-blue-700 transition-colors flex items-center space-x-2"
                      >
                        <Download className="w-5 h-5" />
                        <span>Download PDF</span>
                      </a>
                      <button className="px-6 py-3 bg-gray-600 text-white rounded-xl hover:bg-gray-700 transition-colors flex items-center space-x-2">
                        <FileText className="w-5 h-5" />
                        <span>Export to ERP</span>
                      </button>
                    </div>
                  </div>
                ) : (
                  <div className="text-center py-12 text-gray-500">
                    <FileText className="w-16 h-16 mx-auto mb-4 opacity-30" />
                    <p>No report generated yet. Please complete analysis first.</p>
                  </div>
                )}
              </div>
            )}
          </div>
        </div>
      </div>
      {/* // Add to render function */}
      <div className="fixed bottom-4 right-4 bg-white p-4 rounded-lg shadow-lg border border-gray-200 z-50">
        <h3 className="font-semibold mb-2">Processing Jobs</h3>
        <div className="space-y-2">
          {processingJobs.map(job => (
            <div key={job.id} className="flex items-center">
              <div className={`w-3 h-3 rounded-full mr-2 ${
                job.status === 'completed' ? 'bg-green-500' : 
                job.status === 'failed' ? 'bg-red-500' : 'bg-yellow-500'
              }`}></div>
              <span className="text-sm">
                {job.type.replace('_', ' ')}: {job.status}
              </span>
            </div>
          ))}
        </div>
      </div>

      {/* Footer */}
      <div className="bg-gray-900 text-white py-12">
        <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8">
          <div className="grid md:grid-cols-3 gap-8">
            <div>
              <h3 className="text-lg font-semibold mb-4">FieldServiceAI</h3>
              <p className="text-gray-300 text-sm">
                Transforming field inspections with AI-powered automation. 
                Reduce reporting time by 85% while improving accuracy and compliance.
              </p>
            </div>
            <div>
              <h3 className="text-lg font-semibold mb-4">Key Features</h3>
              <ul className="space-y-2 text-gray-300 text-sm">
                <li>• Computer Vision Object Detection</li>
                <li>• Speech Recognition & NLP</li>
                <li>• Document Analysis & Extraction</li>
                <li>• Automated Report Generation</li>
                <li>• Real-time Compliance Monitoring</li>
              </ul>
            </div>
            <div>
              <h3 className="text-lg font-semibold mb-4">Industries</h3>
              <ul className="space-y-2 text-gray-300 text-sm">
                <li>• Utilities & Power</li>
                <li>• Oil & Gas</li>
                <li>• Telecommunications</li>
                <li>• Manufacturing</li>
                <li>• Construction</li>
              </ul>
            </div>
          </div>
          <div className="border-t border-gray-800 mt-8 pt-8 text-center text-gray-400 text-sm">
            <p>&copy; 2025 FieldServiceAI. Built with React, Hugging Face Transformers, and FastAPI.</p>
            <p className="mt-2">Scalable AI-powered field service automation platform.</p>
          </div>
        </div>
      </div>
    </div>
  );
};

export default FieldServiceAI;