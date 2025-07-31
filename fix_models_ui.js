// Missing function to update the models grid
function updateModelsGrid(models) {
    console.log('📊 Updating models grid with', models.length, 'models');

    const modelsGrid = document.getElementById('models-grid');
    if (!modelsGrid) {
        console.error('❌ Models grid element not found');
        return;
    }

    modelsGrid.innerHTML = models.map(model => `
        <div class="model-card" data-model-name="${model.name}">
            <div class="model-header">
                <h4>${model.name.split('/').pop()}</h4>
                <span class="model-type ${model.type}">${model.type}</span>
            </div>
            <div class="model-info">
                <p>${model.description || 'No description'}</p>
                <div class="model-meta">
                    <span>Size: ${model.size}</span>
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

    console.log('✅ Models grid updated successfully');
}

// Test the API
fetch('/api/models')
    .then(response => response.json())
    .then(data => {
        console.log('📡 API Response:', data);
        if (data.status === 'success') {
            console.log('🎯 Found', data.models.length, 'models:');
            data.models.forEach((model, i) => {
                console.log(`  ${i+1}. ${model.name} (${model.status})`);
            });
        }
    });
