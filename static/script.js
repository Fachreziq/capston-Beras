(function () {
    const RING_CIRCUMFERENCE = 326.7; // 2 * π * r(52)

    // Isi ring gauge (stroke-dashoffset) dan bar probabilitas dari data-pct
    document.querySelectorAll(".ring-fill[data-pct]").forEach(function (el) {
        const pct = parseFloat(el.getAttribute("data-pct")) || 0;
        const offset = RING_CIRCUMFERENCE * (1 - pct / 100);
        requestAnimationFrame(function () {
            el.style.strokeDashoffset = offset;
        });
    });

    document.querySelectorAll(".bar-fill[data-pct]").forEach(function (el) {
        const pct = parseFloat(el.getAttribute("data-pct")) || 0;
        requestAnimationFrame(function () {
            el.style.width = pct + "%";
        });
    });

    const dropzone = document.getElementById("dropzone");
    const input = document.getElementById("image-input");
    const previewWrap = document.getElementById("preview-wrap");
    const previewImg = document.getElementById("preview-img");
    const emptyState = document.getElementById("dropzone-empty");
    const filenameDisplay = document.getElementById("filename-display");
    const submitBtn = document.getElementById("submit-btn");

    function showFile(file) {
        if (!file) return;
        filenameDisplay.textContent = file.name;

        const reader = new FileReader();
        reader.onload = function (e) {
            previewImg.src = e.target.result;
            previewWrap.hidden = false;
            emptyState.hidden = true;
        };
        reader.readAsDataURL(file);
    }

    if (input) {
        input.addEventListener("change", function () {
            if (input.files && input.files[0]) {
                showFile(input.files[0]);
            }
        });
    }

    if (dropzone) {
        ["dragenter", "dragover"].forEach(function (evt) {
            dropzone.addEventListener(evt, function (e) {
                e.preventDefault();
                dropzone.classList.add("drag-over");
            });
        });

        ["dragleave", "drop"].forEach(function (evt) {
            dropzone.addEventListener(evt, function (e) {
                e.preventDefault();
                dropzone.classList.remove("drag-over");
            });
        });

        dropzone.addEventListener("drop", function (e) {
            const file = e.dataTransfer.files[0];
            if (file) {
                input.files = e.dataTransfer.files;
                showFile(file);
            }
        });
    }

    const form = document.getElementById("upload-form");
    if (form) {
        form.addEventListener("submit", function () {
            submitBtn.disabled = true;
            submitBtn.textContent = "Menganalisis…";
        });
    }
})();