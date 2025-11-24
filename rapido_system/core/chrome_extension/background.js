// Background script for tab capture automation
console.log('🚀 Tab Capture Extension Background Script Loaded');

let captureState = {
  isCapturing: false,
  tabId: null,
  stream: null,
  recorder: null,
  chunks: []
};

// Listen for messages from content scripts
chrome.runtime.onMessage.addListener((message, sender, sendResponse) => {
  console.log('📨 Background received message:', message);
  
  switch (message.action) {
    case 'START_TAB_CAPTURE':
      handleStartCapture(message, sender, sendResponse);
      return true; // Keep sendResponse active for async response
      
    case 'STOP_TAB_CAPTURE':
      handleStopCapture(message, sender, sendResponse);
      return true;
      
    case 'GET_CAPTURE_STATUS':
      sendResponse({ 
        isCapturing: captureState.isCapturing,
        tabId: captureState.tabId 
      });
      break;
      
    default:
      console.warn('⚠️ Unknown message action:', message.action);
  }
});

async function handleStartCapture(message, sender, sendResponse) {
  try {
    console.log('🎬 Starting tab capture for tab:', sender.tab?.id);
    
    if (captureState.isCapturing) {
      console.log('⚠️ Capture already in progress');
      sendResponse({ success: false, error: 'Capture already in progress' });
      return;
    }
    
    const tabId = sender.tab?.id;
    if (!tabId) {
      throw new Error('No tab ID available');
    }
    
    // Start tab capture using Chrome's tabCapture API
    const stream = await new Promise((resolve, reject) => {
      chrome.tabCapture.capture({
        video: true,
        audio: message.includeAudio !== false, // Default to true unless explicitly false
        videoConstraints: {
          mandatory: {
            minWidth: 1920,
            maxWidth: 1920,
            minHeight: 1080,
            maxHeight: 1080,
            minFrameRate: 25,
            maxFrameRate: 25
          }
        }
      }, (captureStream) => {
        if (chrome.runtime.lastError) {
          reject(new Error(chrome.runtime.lastError.message));
        } else if (!captureStream) {
          reject(new Error('Failed to capture tab stream'));
        } else {
          resolve(captureStream);
        }
      });
    });
    
    console.log('✅ Tab capture stream obtained:', stream.id);
    
    // Create or get offscreen document for recording
    await ensureOffscreenDocument();
    
    // Send stream to offscreen document for recording
    const recordingResult = await chrome.runtime.sendMessage({
      action: 'START_RECORDING',
      streamId: stream.id,
      duration: message.duration || 30000, // Default 30 seconds
      format: message.format || 'webm'
    });
    
    if (!recordingResult.success) {
      throw new Error(recordingResult.error);
    }
    
    // Update capture state
    captureState = {
      isCapturing: true,
      tabId: tabId,
      stream: stream,
      recorder: null, // Handled in offscreen document
      chunks: []
    };
    
    console.log('🎯 Tab capture started successfully');
    sendResponse({ 
      success: true, 
      streamId: stream.id,
      tabId: tabId,
      message: 'Tab capture started'
    });
    
  } catch (error) {
    console.error('❌ Tab capture failed:', error);
    sendResponse({ 
      success: false, 
      error: error.message 
    });
  }
}

async function ensureOffscreenDocument() {
  // Check if offscreen document already exists
  const existingContexts = await chrome.runtime.getContexts({
    contextTypes: ['OFFSCREEN_DOCUMENT'],
    documentUrls: [chrome.runtime.getURL('offscreen.html')]
  });

  if (existingContexts.length > 0) {
    console.log('📄 Offscreen document already exists');
    return;
  }

  // Create offscreen document
  console.log('📄 Creating offscreen document');
  await chrome.offscreen.createDocument({
    url: 'offscreen.html',
    reasons: ['USER_MEDIA'],
    justification: 'Recording tab capture stream with MediaRecorder'
  });
  
  console.log('✅ Offscreen document created');
}
