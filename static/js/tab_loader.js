// Tab-specific content loader
console.log('📂 Tab loader initialized');

// Add event listeners to tab buttons
document.addEventListener('DOMContentLoaded', function() {
    const navTabs = document.querySelectorAll('.nav-tab');
    
    navTabs.forEach(tab => {
        tab.addEventListener('click', function() {
            const tabName = this.dataset.tab;
            console.log(`📂 Loading content for ${tabName} tab`);
            
            // Wait a moment for tab to become active, then load content
            setTimeout(() => {
                loadTabContent(tabName);
            }, 100);
        });
    });
    
    // Load content for initially active tab
    const activeTab = document.querySelector('.tab-content.active');
    if (activeTab) {
        const tabName = activeTab.id.replace('-tab', '');
        loadTabContent(tabName);
    }
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
            await loadPerformance();
            break;
        case 'settings':
            await loadSettings();
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
        const response = await fetch('/api/performance');
        const data = await response.json();
        if (data.status === 'success') {
            // Update performance metrics
            const tokensEl = document.getElementById('perf-tokens-sec');
            const responseTimeEl = document.getElementById('perf-response-time');
            const memoryEl = document.getElementById('perf-memory');
            const memoryAvailEl = document.getElementById('perf-memory-available');
            const docCountEl = document.getElementById('perf-doc-count');
            const vectorCountEl = document.getElementById('perf-vector-count');
            
            if (tokensEl && data.model_performance) {
                tokensEl.textContent = data.model_performance.tokens_per_second.toFixed(1);
            }
            if (responseTimeEl && data.model_performance) {
                responseTimeEl.textContent = data.model_performance.avg_response_time.toFixed(3) + 's';
            }
            if (memoryEl && data.model_performance) {
                memoryEl.textContent = data.model_performance.memory_usage.used_gb.toFixed(1) + 'GB';
            }
            if (memoryAvailEl && data.model_performance) {
                memoryAvailEl.textContent = data.model_performance.memory_usage.available_gb.toFixed(1) + 'GB';
            }
            if (docCountEl) {
                docCountEl.textContent = data.document_count || 0;
            }
            if (vectorCountEl && data.vector_store) {
                vectorCountEl.textContent = data.vector_store.total_vectors || 0;
            }
            
            console.log('✅ Performance metrics loaded');
        }
    } catch (e) {
        console.error('Failed to load performance metrics:', e);
    }
}

async function loadSettings() {
    console.log('⚙️ Settings tab loaded');
    // Settings are static in the HTML, but we could load current values from API if needed
} 