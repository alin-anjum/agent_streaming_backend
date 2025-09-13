// Offscreen document for MediaRecorder operations
console.log('📄 Offscreen document loaded');

let recorder = null;
let chunks = [];

chrome.runtime.onMessage.addListener((message, sender, sendResponse) => {
  console.log('📨 Offscreen received message:', message);
  
  switch (message.action) {
    case 'START_RECORDING':
      handleStartRecording(message, sender, sendResponse);
      return true;
      
    case 'STOP_RECORDING':
      handleStopRecording(message, sender, sendResponse);
      return true;
      
    default:
      console.warn('⚠️ Unknown offscreen message action:', message.action);
  }
});

async function handleStartRecording(message, sender, sendResponse) {
  try {
    console.log('🎬 Starting recording with stream ID:', message.streamId);
    
    // Get the stream from the stream ID
    const stream = await navigator.mediaDevices.getUserMedia({
      video: { mediaSource: 'tab' }
    });
    
    console.log('✅ Stream obtained in offscreen:', stream);
    
    // Create MediaRecorder
    recorder = new MediaRecorder(stream, {
      mimeType: 'video/webm;codecs=vp9'
    });
    
    chunks = [];
    
    recorder.ondataavailable = (event) => {
      if (event.data.size > 0) {
        chunks.push(event.data);
        console.log('📊 Recorded chunk:', event.data.size, 'bytes');
      }
    };
    
    recorder.onstop = () => {
      console.log('🛑 Recording stopped, creating download...');
      
      const blob = new Blob(chunks, { type: 'video/webm' });
      const url = URL.createObjectURL(blob);
      
      // Create download
      const a = document.createElement('a');
      a.href = url;
      a.download = `tab_capture_${Date.now()}.webm`;
      document.body.appendChild(a);
      a.click();
      document.body.removeChild(a);
      
      URL.revokeObjectURL(url);
      console.log('✅ Download created');
    };
    
    // Start recording
    recorder.start(1000); // Record in 1-second chunks
    console.log('🎯 Recording started');
    
    // Auto-stop after duration
    if (message.duration) {
      setTimeout(() => {
        if (recorder && recorder.state === 'recording') {
          recorder.stop();
        }
      }, message.duration);
    }
    
    sendResponse({ success: true });
    
  } catch (error) {
    console.error('❌ Recording failed:', error);
    sendResponse({ success: false, error: error.message });
  }
}
