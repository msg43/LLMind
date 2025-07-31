// Tab-specific content loader
console.log('📂 Tab loader initialized');

// Note: Tab clicking is now handled by app.js to avoid conflicts
// This file just provides the content loading functions

document.addEventListener('DOMContentLoaded', function() {
    console.log('📂 Tab loader DOMContentLoaded - content loading functions available');
});

async function loadTabContent(tabName) {
    console.log(`🔄 Loading content for tab: ${tabName}`);

    switch(tabName) {
        case 'history':
            await loadChatHistory();
            break;
        case 'documents':
            await loadDocuments();
            break;
        case 'models':
            await loadModels();
            break;
        case 'voice':
            await loadVoices();
            break;
        case 'reasoning':
            await loadReasoning();
            break;
        case 'performance':
            console.log('📊 Performance tab selected, loading data...');
            // Ensure the performance tab is visible
            const perfTab = document.getElementById('performance-tab');
            if (perfTab) {
                perfTab.style.display = 'block';
                perfTab.style.visibility = 'visible';
                perfTab.style.opacity = '1';
                console.log('📊 Performance tab made visible');
            }
            await loadPerformance();
            break;

    }
}

async function loadChatHistory() {
    try {
        const response = await fetch('/api/chats');
        const data = await response.json();
        if (data.status === 'success') {
            const container = document.getElementById('chat-history-main-list');
            if (container) {
                if (data.chats && data.chats.length > 0) {
                    container.innerHTML = data.chats.map(chat => `
                        <div class="chat-history-item" data-chat-id="${chat.id}">
                            <div class="chat-actions">
                                <button class="chat-action-btn export" title="Export Chat" data-action="export" data-chat-id="${chat.id}">
                                    <i class="fas fa-download"></i>
                                </button>
                                <button class="chat-action-btn delete" title="Delete Chat" data-action="delete" data-chat-id="${chat.id}">
                                    <i class="fas fa-trash"></i>
                                </button>
                            </div>
                            <div class="chat-title">${chat.title || 'Untitled'}</div>
                            <div class="chat-preview">${chat.preview || 'No preview'}</div>
                            <div class="chat-meta">
                                <span class="chat-date">${new Date(chat.updated_at).toLocaleDateString()}</span>
                                <span class="chat-message-count">${chat.message_count || 0} msgs</span>
                            </div>
                        </div>
                    `).join('');
                } else {
                    container.innerHTML = `
                        <div class="no-chats">
                            <i class="fas fa-comments"></i>
                            <span>No chats yet</span>
                            <small>Start a new conversation!</small>
                        </div>
                    `;
                }
                console.log('✅ Chat history loaded');
            }
        }
    } catch (e) {
        console.error('Failed to load chats:', e);
    }
}

async function loadDocuments() {
    try {
        const response = await fetch('/api/documents');
        const data = await response.json();
        if (data.status === 'success') {
            const container = document.getElementById('documents-container');
            if (container) {
                if (data.documents && data.documents.length > 0) {
                    container.innerHTML = data.documents.map(doc => `
                        <div class="document-item">
                            <div class="document-info">
                                <h4>${doc.filename || 'Unknown'}</h4>
                                <p>${doc.chunk_count || 0} chunks • ${doc.text_length || 0} characters</p>
                                <span class="extension">${doc.extension || 'unknown'}</span>
                            </div>
                            <button class="btn btn-secondary delete-doc" data-hash="${doc.file_hash || ''}">
                                <i class="fas fa-trash"></i>
                            </button>
                        </div>
                    `).join('');
                } else {
                    container.innerHTML = '<p>No documents uploaded yet.</p>';
                }
                console.log('✅ Documents loaded');
            }
        }
    } catch (e) {
        console.error('Failed to load documents:', e);
    }
}

async function loadModels() {
    try {
        const response = await fetch('/api/models');
        const data = await response.json();
        if (data.status === 'success') {
            // Update current model info
            if (data.current_model) {
                const nameEl = document.getElementById('current-model-name');
                const statusEl = document.getElementById('current-model-status');
                if (nameEl) nameEl.textContent = data.current_model.name || 'None';
                if (statusEl) statusEl.textContent = data.current_model.status || 'Not loaded';
            }

            // Update available models grid if it's empty (server-side rendering might not have loaded)
            const modelsGrid = document.getElementById('models-grid');
            if (modelsGrid && (!modelsGrid.innerHTML || modelsGrid.innerHTML.trim() === '')) {
                if (data.models && data.models.length > 0) {
                    modelsGrid.innerHTML = data.models.map(model => `
                        <div class="model-card" data-model-name="${model.name}">
                            <div class="model-header">
                                <h4>${model.name.split('/').pop()}</h4>
                                <span class="model-type ${model.type || ''}">${model.type || 'general'}</span>
                            </div>
                            <div class="model-info">
                                <p>${model.description || 'No description'}</p>
                                <div class="model-meta">
                                    <span>Size: ${model.size || 'Unknown'}</span>
                                    <span class="status ${model.status}">${model.status}</span>
                                </div>
                            </div>
                            <div class="model-actions">
                                ${model.status === 'downloaded' ?
                                    `<button class="btn btn-primary switch-model" data-model="${model.name}">
                                        <i class="fas fa-play"></i> Use Model
                                    </button>` :
                                    `<button class="btn btn-secondary download-model" data-model="${model.name}">
                                        <i class="fas fa-download"></i> Download
                                    </button>`
                                }
                            </div>
                        </div>
                    `).join('');
                }
            }

            console.log('✅ Model info updated');
        }
    } catch (e) {
        console.error('Failed to load models:', e);
    }
}

async function loadVoices() {
    console.log('🎤 Voice tab - no dynamic content to load');
}

async function loadReasoning() {
    console.log('🧠 Reasoning tab - no dynamic content to load');
}

async function loadPerformance() {
    try {
        console.log('📊 Loading performance data...');
        const response = await fetch('/api/performance');
        const data = await response.json();

        if (data.status === 'success') {
            // Update model performance metrics
            updateElement('perf-tokens-sec', data.model_performance?.tokens_per_second?.toFixed(1) || '--');
            updateElement('perf-response-time', data.model_performance?.avg_response_time?.toFixed(3) + 's' || '--');
            updateElement('perf-gpu', data.model_performance?.gpu_utilization?.toFixed(1) + '%' || '--');

            // Update memory usage
            const memUsage = data.model_performance?.memory_usage || {};
            updateElement('perf-memory-used', memUsage.used_gb ? memUsage.used_gb.toFixed(1) + 'GB' : '--');
            updateElement('perf-memory-available', memUsage.total_gb ? (memUsage.total_gb - memUsage.used_gb).toFixed(1) + 'GB' : '--');
            updateElement('perf-memory-peak', memUsage.used_gb ? memUsage.used_gb.toFixed(1) + 'GB' : '--');

            // Update system stats
            updateElement('perf-doc-count', data.document_count || '0');
            updateElement('perf-vector-count', data.vector_store?.total_vectors || '0');
            updateElement('perf-conversation-count', data.chat_performance?.total_conversations || '0');

            // Update chat performance
            updateElement('perf-chat-response', data.chat_performance?.avg_response_time?.toFixed(3) + 's' || '--');
            updateElement('perf-total-messages', data.chat_performance?.total_conversations ? (data.chat_performance.total_conversations * 2) : '0');

            // Get current model info
            const modelResponse = await fetch('/api/status');
            const modelData = await modelResponse.json();
            updateElement('perf-active-model', modelData.model?.name?.split('/').pop() || 'None');

            console.log('✅ Performance metrics loaded successfully');
        } else {
            console.error('Failed to load performance data:', data.message);
        }
    } catch (e) {
        console.error('Failed to load performance metrics:', e);
        // Set all values to error state
        const perfElements = [
            'perf-tokens-sec', 'perf-response-time', 'perf-gpu',
            'perf-memory-used', 'perf-memory-available', 'perf-memory-peak',
            'perf-doc-count', 'perf-vector-count', 'perf-conversation-count',
            'perf-chat-response', 'perf-total-messages', 'perf-active-model'
        ];
        perfElements.forEach(id => updateElement(id, 'Error'));
    }
}

function updateElement(id, value) {
    const element = document.getElementById(id);
    if (element) {
        element.textContent = value;
    } else {
        console.warn(`Element not found: ${id}`);
    }
}

// Simple test function to manually check performance loading
window.debugPerformance = function() {
    console.log('🔍 Debug: Testing performance elements...');

    // Test if elements exist
    const testElements = [
        'perf-tokens-sec', 'perf-response-time', 'perf-gpu',
        'perf-memory-used', 'perf-memory-available', 'perf-memory-peak',
        'perf-doc-count', 'perf-vector-count', 'perf-conversation-count'
    ];

    testElements.forEach(id => {
        const el = document.getElementById(id);
        console.log(`Element ${id}:`, el ? 'EXISTS' : 'MISSING');
    });

    // Test API directly
    fetch('/api/performance')
        .then(r => r.json())
        .then(data => {
            console.log('API Response:', data);
            if (data.status === 'success') {
                console.log('✅ API working, calling loadPerformance...');
                loadPerformance();
            }
        })
        .catch(e => console.error('API Error:', e));
}

// Force show performance tab and load data
window.showPerformanceTab = function() {
    console.log('🔧 Forcing performance tab to show...');

    // Hide all other tabs
    document.querySelectorAll('.tab-content').forEach(tab => {
        tab.classList.remove('active');
        tab.style.display = 'none';
    });

    // Show performance tab
    const perfTab = document.getElementById('performance-tab');
    if (perfTab) {
        perfTab.classList.add('active');
        perfTab.style.display = 'block';
        perfTab.style.visibility = 'visible';
        perfTab.style.opacity = '1';
        console.log('✅ Performance tab forced visible');

        // Load data
        loadPerformance();
    } else {
        console.error('❌ Performance tab not found!');
    }
}
