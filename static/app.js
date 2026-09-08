const input = document.getElementById('imageInput');
const preview = document.getElementById('clientPreview');
const prompt = document.getElementById('dropPrompt');
const zone = document.getElementById('dropZone');
const form = document.getElementById('predictionForm');
const button = document.getElementById('analyzeButton');

function previewFile(file) {
    if (!file || !file.type.startsWith('image/')) return;
    const reader = new FileReader();
    reader.onload = event => {
        preview.src = event.target.result;
        preview.classList.remove('hidden');
        prompt.classList.add('hidden');
        zone.classList.add('has-image');
    };
    reader.readAsDataURL(file);
}

if (input) input.addEventListener('change', () => previewFile(input.files[0]));
if (zone) {
    ['dragenter', 'dragover'].forEach(name => zone.addEventListener(name, event => { event.preventDefault(); zone.classList.add('dragging'); }));
    ['dragleave', 'drop'].forEach(name => zone.addEventListener(name, event => { event.preventDefault(); zone.classList.remove('dragging'); }));
    zone.addEventListener('drop', event => {
        if (!event.dataTransfer.files.length) return;
        const transfer = new DataTransfer();
        transfer.items.add(event.dataTransfer.files[0]);
        input.files = transfer.files;
        previewFile(input.files[0]);
    });
}
if (form) form.addEventListener('submit', () => {
    button.disabled = true;
    button.innerHTML = '<span class="spinner"></span><span>Analyzing image…</span>';
});
