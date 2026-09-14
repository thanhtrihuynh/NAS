const state = {
    records: [],
    record: null,
    waveform: null,
    segment: null,
    segmentMode: "raw",
    currentNas: null,
    candidates: [],
    candidateJobs: [],
    candidateJobId: null,
    selectedCandidateId: null,
    bestCandidateId: null,
    activeCandidateIndex: -1,
    inferenceMode: "local",
    fpgaResources: null,
    fullTestJobId: null,
    fullTestPollTimer: null,
    nasPollTimer: null,
    classFilter: "",
    segments: [],
    beatSource: "atr",
    status: null,

    // ECG record browser
    recordWindow: null,
    windowSeconds: 10,
    windowStartSec: 0,
    windowFetchTimer: null,
    windowRequestSerial: 0,
};

const $ = (id) => document.getElementById(id);

function formatApiError(detail, status) {
    if (typeof detail === "string") {
        return detail;
    }

    if (Array.isArray(detail)) {
        return detail.map(item => {
            if (typeof item === "string") {
                return item;
            }

            const location = Array.isArray(item?.loc)
                ? item.loc.join(" → ")
                : "";

            const message = item?.msg || JSON.stringify(item);

            return location
                ? `${location}: ${message}`
                : message;
        }).join(" | ");
    }

    if (detail && typeof detail === "object") {
        if (detail.message) {
            return String(detail.message);
        }

        if (detail.detail) {
            return formatApiError(detail.detail, status);
        }

        return JSON.stringify(detail, null, 2);
    }

    return `HTTP ${status}`;
}



async function api(url, options = {}) {
    const response = await fetch(
        url,
        options
    );

    // A Fetch Response body can only be consumed once.
    // Read it once as text, then parse JSON from that string.
    const raw = await response.text();

    let data = null;

    if (raw) {
        try {
            data = JSON.parse(raw);
        }
        catch {
            data = {
                detail: raw
            };
        }
    }
    else {
        data = {};
    }

    if (!response.ok) {
        throw new Error(
            formatApiError(
                data?.detail ?? data,
                response.status
            )
        );
    }

    return data;
}


function escapeHtml(value) {
    const div = document.createElement("div");
    div.textContent = String(value);
    return div.innerHTML;
}

function setBox(element, text, mode = "") {
    element.textContent = text;
    element.classList.remove("success", "error", "warn");
    if (mode) element.classList.add(mode);
}

function formatNumber(value, digits = 3) {
    if (value === null || value === undefined || value === "") return "-";
    const number = Number(value);
    if (Number.isNaN(number)) return String(value);
    return number.toLocaleString(undefined, { maximumFractionDigits: digits });
}

function formatShape(shape) {
    if (!shape) return "-";
    return Array.isArray(shape) ? shape.join(" × ") : String(shape);
}

function updateDateTime() {
    const now = new Date();
    $("dateText").textContent = now.toISOString().slice(0, 10);
    $("timeText").textContent = now.toLocaleTimeString();
}


async function loadStatus() {
    const data = await api("/api/status");

    $("serverDot").classList.add("online");
    $("serverText").textContent = "Server Online";

    $("datasetStatus").textContent =
        `${data.records_found} ECG records found in datasets`
        + ` · test rows: ${data.test_samples}`;
}




async function loadRecords() {
    const params = new URLSearchParams();

    if (state.classFilter) {
        params.set(
            "label",
            state.classFilter
        );
    }

    $("recordSelect").disabled = true;
    $("channelSelect").disabled = true;
    $("segmentSelect").disabled = true;

    const data = await api(
        `/api/records?${params.toString()}`
    );

    state.records = data.records || [];

    const select = $("recordSelect");
    select.innerHTML = "";

    if (state.records.length === 0) {
        const option = document.createElement("option");
        option.value = "";
        option.textContent = "No ECG records found";
        select.appendChild(option);

        $("recordInfo").textContent =
            "Không tìm thấy record .dat + .hea trong thư mục datasets.";

        $("datasetError").textContent =
            "Kiểm tra http://127.0.0.1:8000/api/debug/dataset";

        $("runInferenceButton").disabled = true;
        return;
    }

    for (const record of state.records) {
        const option = document.createElement("option");

        option.value = record.record_id;

        option.textContent =
            `${record.filename}`
            + (record.has_atr ? " · ATR" : "")
            + (record.test_segments > 0
                ? ` · ${record.test_segments} test beats`
                : "");

        select.appendChild(option);
    }

    select.disabled = false;

    await selectRecord(
        state.records[0].record_id
    );
}



async function selectRecord(recordId) {
    if (!recordId) {
        return;
    }

    const record = state.records.find(
        item =>
            String(item.record_id)
            === String(recordId)
    );

    state.record = record || null;

    if (!record) {
        return;
    }

    $("recordMetric").textContent =
        `Record: ${record.filename}`;

    $("recordTitle").textContent =
        `ECG Record Browser (${record.filename})`;

    const channelSelect =
        $("channelSelect");

    channelSelect.innerHTML = "";

    const channels =
        record.channels || [];

    if (channels.length === 0) {
        const option =
            document.createElement(
                "option"
            );

        option.value = "";
        option.textContent =
            "No channel found";

        channelSelect.appendChild(
            option
        );

        channelSelect.disabled = true;
    }
    else {
        for (const channel of channels) {
            const option =
                document.createElement(
                    "option"
                );

            option.value = channel;
            option.textContent = channel;

            channelSelect.appendChild(
                option
            );
        }

        channelSelect.disabled = false;
    }

    $("channelMetric").textContent =
        `Channel: ${channelSelect.value || "-"}`;

    $("durationMetric").textContent =
        record.fs && record.sig_len
            ? `Duration: ${(record.sig_len / record.fs / 60).toFixed(1)} min`
            : "Duration: -";

    $("recordInfo").textContent =
        `Record ID: ${record.record_id}\n`
        + `File: ${record.filename}\n`
        + `Sampling: ${record.fs ?? "-"} Hz\n`
        + `Channels: ${channels.join(", ") || "-"}\n`
        + `ATR: ${record.has_atr ? "available" : "missing"}\n`
        + `Test beats: ${record.test_segments}`;

    await loadRecordWaveform(
        recordId,
        channelSelect.value || null
    );

    await loadSegments(
        recordId
    );
}



async function loadSegments(recordId) {
    const params =
        new URLSearchParams();

    params.set(
        "source",
        state.beatSource || "atr"
    );

    if (state.classFilter) {
        params.set(
            "label",
            state.classFilter
        );
    }

    const data = await api(
        `/api/records/${encodeURIComponent(recordId)}/beats?${params.toString()}`
    );

    state.segments =
        data.beats || [];

    const select =
        $("segmentSelect");

    select.innerHTML = "";

    if (state.segments.length === 0) {
        const option =
            document.createElement(
                "option"
            );

        option.value = "";

        option.textContent =
            state.beatSource === "test"
                ? "No test-set beat in this record"
                : "No N/L/R/V/A annotation";

        select.appendChild(
            option
        );

        select.disabled = true;

        state.segment = null;

        $("runInferenceButton").disabled = true;

        setBox(
            $("inferenceState"),
            state.beatSource === "test"
                ? "Không có beat phù hợp trong test.csv. Chuyển Beat Source sang All annotated beats (.atr)."
                : "Không có annotation N/L/R/V/A trong file .atr.",
            "warn"
        );

        return;
    }

    state.segments.forEach(
        (beat, i) => {
            const option =
                document.createElement(
                    "option"
                );

            option.value =
                beat.token;

            option.textContent =
                `Beat ${i + 1}`
                + ` · ${beat.label}`
                + ` · sample ${beat.center}`;

            select.appendChild(
                option
            );
        }
    );

    select.disabled = false;

    await loadSegment(
        state.segments[0].token,
        false
    );
}



async function loadRecordWaveform(
    recordId,
    channel
) {
    const params =
        new URLSearchParams();

    if (channel) {
        params.set(
            "channel",
            channel
        );
    }

    // Overview only. Detailed ECG is fetched separately without
    // downsampling through /window.
    params.set(
        "max_points",
        "7000"
    );

    const data = await api(
        `/api/records/${encodeURIComponent(recordId)}/waveform?${params.toString()}`
    );

    state.waveform = data;

    $("channelMetric").textContent =
        `Channel: ${data.channel}`;

    $("durationMetric").textContent =
        `Duration: ${(data.duration_seconds / 60).toFixed(1)} min`;

    renderMarkerLegend(
        data.markers
    );

    configureRecordSlider();

    drawOverviewWaveform();

    await loadRecordWindow(
        state.windowStartSec
    );
}


function renderMarkerLegend(markers) {
    const counts = {};

    for (
        const marker
        of markers || []
    ) {
        counts[marker.label] =
            (counts[marker.label] || 0)
            + 1;
    }

    $("markerLegend").textContent =
        Object.entries(counts)
            .map(
                ([label, count]) =>
                    `${label}: ${count}`
            )
            .join(" · ")
        || "No annotated beats";
}


function formatClock(seconds) {
    const value = Math.max(
        0,
        Number(seconds) || 0
    );

    const whole = Math.floor(
        value
    );

    const minutes = Math.floor(
        whole / 60
    );

    const secs =
        whole % 60;

    return (
        `${minutes}:`
        + `${String(secs).padStart(2, "0")}`
    );
}



function safeWindowSeconds() {
    const selectValue =
        Number(
            $("windowSecondsSelect")?.value
        );

    const stateValue =
        Number(
            state.windowSeconds
        );

    let value =
        Number.isFinite(stateValue)
            ? stateValue
            : selectValue;

    if (!Number.isFinite(value)) {
        value = 10;
    }

    value =
        Math.max(
            1,
            Math.min(
                60,
                value
            )
        );

    state.windowSeconds = value;

    return value;
}


function configureRecordSlider() {
    const data = state.waveform;

    if (!data) {
        return;
    }

    const totalDuration =
        Number(
            data.duration_seconds
        );

    if (
        !Number.isFinite(totalDuration)
        || totalDuration <= 0
    ) {
        return;
    }

    const duration =
        safeWindowSeconds();

    const slider =
        $("recordPositionSlider");

    const maxStart =
        Math.max(
            0,
            totalDuration - duration
        );

    let start =
        Number(
            state.windowStartSec
        );

    if (!Number.isFinite(start)) {
        start = 0;
    }

    start =
        Math.max(
            0,
            Math.min(
                start,
                maxStart
            )
        );

    state.windowStartSec = start;

    slider.min = "0";
    slider.max =
        String(maxStart);

    slider.step = "0.25";

    slider.value =
        String(start);

    $("sliderEndTime").textContent =
        formatClock(
            totalDuration
        );

    updateSliderTimeText();
}


function updateSliderTimeText() {
    $("sliderCurrentTime").textContent =
        formatClock(
            state.windowStartSec
        );
}



async function loadRecordWindow(
    startSec = state.windowStartSec
) {
    if (
        !state.record
        || !state.waveform
    ) {
        return;
    }

    const duration =
        safeWindowSeconds();

    const totalDuration =
        Number(
            state.waveform.duration_seconds
        );

    if (
        !Number.isFinite(totalDuration)
        || totalDuration <= 0
    ) {
        throw new Error(
            "Invalid ECG record duration."
        );
    }

    const maxStart =
        Math.max(
            0,
            totalDuration - duration
        );

    let requestedStart =
        Number(startSec);

    if (!Number.isFinite(requestedStart)) {
        requestedStart = 0;
    }

    requestedStart =
        Math.max(
            0,
            Math.min(
                requestedStart,
                maxStart
            )
        );

    state.windowStartSec =
        requestedStart;

    const params =
        new URLSearchParams();

    params.set(
        "channel",
        $("channelSelect").value
        || state.waveform.channel
    );

    params.set(
        "start_sec",
        requestedStart.toFixed(6)
    );

    params.set(
        "duration_sec",
        duration.toFixed(6)
    );

    const requestSerial =
        ++state.windowRequestSerial;

    const data = await api(
        `/api/records/${encodeURIComponent(state.record.record_id)}/window?${params.toString()}`
    );

    // Ignore a slower response from an older slider position.
    if (
        requestSerial
        !== state.windowRequestSerial
    ) {
        return;
    }

    state.recordWindow = data;

    state.windowStartSec =
        Number.isFinite(
            Number(data.start_sec)
        )
            ? Number(data.start_sec)
            : requestedStart;

    $("recordPositionSlider").value =
        String(
            state.windowStartSec
        );

    updateSliderTimeText();

    $("windowRangeText").textContent =
        `Showing ${formatClock(data.start_sec)} – ${formatClock(data.end_sec)}`
        + ` · ${Number(data.duration_sec).toFixed(1)} s`;

    $("windowBeatCount").textContent =
        `${data.markers.length} beat${data.markers.length === 1 ? "" : "s"}`;

    drawWindowWaveform();
    drawOverviewWaveform();
}


function scheduleWindowLoad(
    startSec
) {
    state.windowStartSec =
        Number(startSec) || 0;

    updateSliderTimeText();
    drawOverviewWaveform();

    if (state.windowFetchTimer) {
        clearTimeout(
            state.windowFetchTimer
        );
    }

    state.windowFetchTimer =
        setTimeout(
            () => {
                loadRecordWindow(
                    state.windowStartSec
                ).catch(
                    error => {
                        console.error(
                            error
                        );
                    }
                );
            },
            100
        );
}



async function centerWindowOnBeat(
    sampleCenter
) {
    if (
        !state.waveform
        || !state.waveform.fs
    ) {
        return;
    }

    const center =
        Number(sampleCenter);

    const fs =
        Number(
            state.waveform.fs
        );

    if (
        !Number.isFinite(center)
        || !Number.isFinite(fs)
        || fs <= 0
    ) {
        return;
    }

    const duration =
        safeWindowSeconds();

    const beatSec =
        center / fs;

    const targetStart =
        Math.max(
            0,
            beatSec
            - duration / 2
        );

    await loadRecordWindow(
        targetStart
    );
}


async function loadSegment(
    beatToken,
    autoPredict = true
) {
    if (!beatToken) {
        return;
    }

    const sample = await api(
        `/api/beats/${encodeURIComponent(beatToken)}`
    );

    state.segment = sample;

    $("segmentInfoGrid").innerHTML = [
        ["Record", sample.filename],
        ["True Label", sample.label],
        ["Source", sample.source],
        ["Channel", sample.channel],
        ["Center", sample.center],
        ["Tensor", sample.shape.join(" × ")],
    ]
        .map(([label, value]) => `
            <div class="info-cell">
                <span>${escapeHtml(label)}</span>
                <strong>${escapeHtml(value)}</strong>
            </div>
        `)
        .join("");

    $("runInferenceButton").disabled = false;

    setBox(
        $("inferenceState"),
        "Heartbeat loaded. Select Local Integer or PYNQ FPGA Hybrid."
    );

    drawSegmentWaveform();

    await centerWindowOnBeat(
        sample.center
    );

    if (autoPredict) {
        updateExecutionModeUi();
    }
}


function getCanvasContext(canvas) {
    const rect = canvas.getBoundingClientRect();
    const width = Math.max(300, rect.width);
    const height = Math.max(70, rect.height);
    const dpr = Math.min(window.devicePixelRatio || 1, 2);
    canvas.width = Math.round(width * dpr);
    canvas.height = Math.round(height * dpr);
    const ctx = canvas.getContext("2d");
    ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
    return { ctx, width, height };
}

function drawGrid(ctx, width, height) {
    ctx.strokeStyle = "rgba(95,126,163,0.16)";
    ctx.lineWidth = 1;
    for (let x = 0; x <= width; x += 45) {
        ctx.beginPath(); ctx.moveTo(x, 0); ctx.lineTo(x, height); ctx.stroke();
    }
    for (let y = 0; y <= height; y += 35) {
        ctx.beginPath(); ctx.moveTo(0, y); ctx.lineTo(width, y); ctx.stroke();
    }
}

function drawSeries(canvas, values, color = "#49a2ff") {
    const { ctx, width, height } = getCanvasContext(canvas);
    ctx.clearRect(0, 0, width, height);
    ctx.fillStyle = "#0a111b";
    ctx.fillRect(0, 0, width, height);
    drawGrid(ctx, width, height);
    if (!values || values.length === 0) return { ctx, width, height };
    let min = Math.min(...values), max = Math.max(...values);
    if (Math.abs(max - min) < 1e-9) { min -= 1; max += 1; }
    const pad = 28;
    const innerW = width - pad * 2;
    const innerH = height - pad * 2;
    ctx.beginPath();
    values.forEach((value, i) => {
        const x = pad + (i / Math.max(1, values.length - 1)) * innerW;
        const normalized = (value - min) / (max - min);
        const y = pad + (1 - normalized) * innerH;
        if (i === 0) ctx.moveTo(x, y); else ctx.lineTo(x, y);
    });
    ctx.strokeStyle = color;
    ctx.lineWidth = 1.6;
    ctx.stroke();
    ctx.fillStyle = "#90a7bf";
    ctx.font = "11px sans-serif";
    ctx.fillText(`max ${max.toFixed(3)}`, 7, 14);
    ctx.fillText(`min ${min.toFixed(3)}`, 7, height - 7);
    return { ctx, width, height, min, max, pad, innerW, innerH };
}


function labelColor(label) {
    const colorMap = {
        N: "#52d273",
        L: "#5b95ff",
        R: "#ffb646",
        V: "#ff5d5d",
        A: "#b56cff",
    };

    return (
        colorMap[label]
        || "#9fb4cb"
    );
}


function drawWindowWaveform() {
    const data =
        state.recordWindow;

    if (!data) {
        return;
    }

    const canvas =
        $("windowCanvas");

    const drawn =
        drawSeries(
            canvas,
            data.values,
            "#43a5ff"
        );

    if (!drawn.ctx) {
        return;
    }

    const {
        ctx,
        width,
        height,
        pad,
        innerW,
    } = drawn;

    // Time axis.
    ctx.fillStyle = "#91a9c3";
    ctx.font = "11px sans-serif";

    const ticks = 5;

    for (
        let i = 0;
        i <= ticks;
        i++
    ) {
        const ratio =
            i / ticks;

        const x =
            pad
            + ratio * innerW;

        const sec =
            data.start_sec
            + ratio * data.duration_sec;

        ctx.fillText(
            formatClock(sec),
            Math.max(
                2,
                x - 14
            ),
            height - 7
        );
    }

    // All beats inside this exact raw window.
    for (
        const marker
        of data.markers || []
    ) {
        const relative =
            marker.sample
            - data.start_sample;

        const sampleCount =
            Math.max(
                1,
                data.end_sample
                - data.start_sample
            );

        const x =
            pad
            + (
                relative
                / sampleCount
            )
            * innerW;

        const selected =
            state.segment
            && String(marker.token)
                === String(
                    state.segment.token
                );

        const color =
            selected
                ? "#ffffff"
                : labelColor(
                    marker.label
                );

        // Short marker only, not a full-height wall.
        ctx.strokeStyle = color;
        ctx.lineWidth =
            selected
                ? 2.4
                : 1.4;

        ctx.beginPath();
        ctx.moveTo(
            x,
            18
        );
        ctx.lineTo(
            x,
            34
        );
        ctx.stroke();

        ctx.fillStyle = color;

        ctx.beginPath();
        ctx.arc(
            x,
            13,
            selected
                ? 5
                : 3.5,
            0,
            Math.PI * 2
        );
        ctx.fill();

        ctx.font =
            selected
                ? "bold 10px sans-serif"
                : "9px sans-serif";

        ctx.fillText(
            marker.label,
            x + 5,
            16
        );
    }
}


function drawOverviewWaveform() {
    const data =
        state.waveform;

    if (!data) {
        return;
    }

    const canvas =
        $("overviewCanvas");

    const {
        ctx,
        width,
        height,
    } = getCanvasContext(
        canvas
    );

    ctx.clearRect(
        0,
        0,
        width,
        height
    );

    ctx.fillStyle = "#08111b";

    ctx.fillRect(
        0,
        0,
        width,
        height
    );

    const padX = 10;
    const top = 12;
    const bottom = 24;

    const innerW =
        width
        - padX * 2;

    const innerH =
        height
        - top
        - bottom;

    const values =
        data.values || [];

    if (values.length > 0) {
        let min =
            Math.min(...values);

        let max =
            Math.max(...values);

        if (
            Math.abs(
                max - min
            )
            < 1e-9
        ) {
            min -= 1;
            max += 1;
        }

        ctx.beginPath();

        values.forEach(
            (value, i) => {
                const x =
                    padX
                    + (
                        i
                        / Math.max(
                            1,
                            values.length - 1
                        )
                    )
                    * innerW;

                const y =
                    top
                    + (
                        1
                        - (
                            value - min
                        )
                        / (
                            max - min
                        )
                    )
                    * innerH;

                if (i === 0) {
                    ctx.moveTo(
                        x,
                        y
                    );
                }
                else {
                    ctx.lineTo(
                        x,
                        y
                    );
                }
            }
        );

        ctx.strokeStyle =
            "rgba(67,165,255,.8)";

        ctx.lineWidth = 1;
        ctx.stroke();
    }

    const totalSamples =
        Math.max(
            1,
            data.signal_length - 1
        );

    // All heartbeat annotations are retained in the overview,
    // but rendered as tiny dots instead of full-height lines.
    for (
        const marker
        of data.markers || []
    ) {
        const x =
            padX
            + (
                marker.sample
                / totalSamples
            )
            * innerW;

        ctx.fillStyle =
            labelColor(
                marker.label
            );

        ctx.fillRect(
            x,
            height - 15,
            1.5,
            5
        );
    }

    // Current viewing window.
    const totalDuration =
        Math.max(
            0.001,
            data.duration_seconds
        );

    const startRatio =
        state.windowStartSec
        / totalDuration;

    const widthRatio =
        Math.min(
            1,
            state.windowSeconds
            / totalDuration
        );

    const selectionX =
        padX
        + startRatio
        * innerW;

    const selectionW =
        Math.max(
            8,
            widthRatio
            * innerW
        );

    ctx.fillStyle =
        "rgba(71,163,255,.15)";

    ctx.fillRect(
        selectionX,
        3,
        selectionW,
        height - 23
    );

    ctx.strokeStyle =
        "#8bc6ff";

    ctx.lineWidth = 2;

    ctx.strokeRect(
        selectionX,
        3,
        selectionW,
        height - 23
    );

    ctx.fillStyle =
        "#91a9c3";

    ctx.font =
        "10px sans-serif";

    ctx.fillText(
        "0:00",
        padX,
        height - 3
    );

    const endText =
        formatClock(
            data.duration_seconds
        );

    ctx.fillText(
        endText,
        Math.max(
            padX,
            width
            - 42
        ),
        height - 3
    );
}


function drawSegmentWaveform() {
    const sample = state.segment;
    if (!sample) return;
    const canvas = $("segmentCanvas");
    const values = state.segmentMode === "raw" ? sample.raw : sample.normalized;
    const color = state.segmentMode === "raw" ? "#4d9cff" : "#b167ff";
    const drawn = drawSeries(canvas, values, color);
    if (!drawn.ctx) return;
    if (state.segmentMode === "normalized") {
        const raw = sample.raw;
        const { ctx, width, height, pad, innerW, innerH } = drawn;
        let min = Math.min(...raw), max = Math.max(...raw);
        if (Math.abs(max - min) < 1e-9) { min -= 1; max += 1; }
        ctx.beginPath();
        raw.forEach((value, i) => {
            const x = pad + (i / Math.max(1, raw.length - 1)) * innerW;
            const normalized = (value - min) / (max - min);
            const y = pad + (1 - normalized) * innerH;
            if (i === 0) ctx.moveTo(x, y); else ctx.lineTo(x, y);
        });
        ctx.strokeStyle = "rgba(77,156,255,0.7)";
        ctx.lineWidth = 1.1;
        ctx.stroke();
    }
    const duration = sample.time[sample.time.length - 1];
    drawn.ctx.fillStyle = "#90a7bf";
    drawn.ctx.font = "11px sans-serif";
    drawn.ctx.fillText("0 s", 28, drawn.height - 6);
    drawn.ctx.fillText(`${duration.toFixed(3)} s`, drawn.width - 60, drawn.height - 6);
}


function updateExecutionModeUi() {
    const localMode = state.inferenceMode === "local";
    $("modeLocalButton").classList.toggle("active", localMode);
    $("modeFpgaButton").classList.toggle("active", !localMode);
    $("runInferenceButton").textContent = localMode
        ? "Run Local Integer Inference"
        : "Run PYNQ FPGA Hybrid Inference";

    const fullTestButton = $("runFullTestButton");

    if (fullTestButton) {
        fullTestButton.textContent = localMode
            ? "Run Full Test Set · Local Integer"
            : "Run Full Test Set · PYNQ FPGA Hybrid";
    }

    $("fpgaResourcePanel").classList.toggle("hidden", localMode);
    $("pynqDebugDetails").classList.toggle("hidden", localMode);
}

function setExecutionMode(mode) {
    state.inferenceMode = mode === "fpga" ? "fpga" : "local";
    updateExecutionModeUi();
    setBox($("inferenceState"), state.segment
        ? (state.inferenceMode === "fpga" ? "PYNQ FPGA Hybrid selected. Ready to run." : "Local Integer selected. Ready to run.")
        : "Select a heartbeat first.");
}

function formatLatency(value) {
    const n = Number(value);
    return Number.isFinite(n) ? `${n.toFixed(3)} ms` : "-";
}

function updateExecutionSummary(result, mode) {
    $("executionSummary").classList.remove("hidden");
    const isFpga = mode === "fpga";
    $("executionModeName").textContent = isFpga ? "PYNQ FPGA Hybrid" : "Local Integer";
    $("executionSourceBadge").textContent = result.source || (isFpga ? "PYNQ_FPGA_HYBRID" : "LAPTOP_INTEGER");
    const timing=result.timing||{};
    if (isFpga) {
        const total=Number(timing.total_hybrid_ms), fpga=Number(timing.fpga_weighted_ms);
        const overhead=Number.isFinite(total)&&Number.isFinite(fpga)?Math.max(0,total-fpga):NaN;
        $("totalLatencyValue").textContent=formatLatency(total);
        $("fpgaLatencyValue").textContent=formatLatency(fpga);
        $("runtimeOverheadValue").textContent=formatLatency(overhead);
        $("fpgaClockValue").textContent=`${Number(state.fpgaResources?.clock_mhz||40).toFixed(0)} MHz`;
    } else {
        $("totalLatencyValue").textContent=formatLatency(timing.total_local_ms);
        $("fpgaLatencyValue").textContent="Not used";
        $("runtimeOverheadValue").textContent="Laptop CPU";
        $("fpgaClockValue").textContent="Not used";
    }
}

function resourceCard(name,r) {
    const actual=r?.used!==null&&r?.used!==undefined;
    const pct=actual&&r?.percent!==null&&r?.percent!==undefined?Number(r.percent):NaN;
    const pctText=Number.isFinite(pct)?`${pct.toFixed(2)}%`:"Actual unavailable";
    const width=Number.isFinite(pct)?Math.max(0,Math.min(100,pct)):0;
    return `<div class="fpga-resource-card">
        <div class="fpga-resource-card-head"><strong>${escapeHtml(name)}</strong><span>${escapeHtml(pctText)}</span></div>
        <div class="fpga-resource-value">${escapeHtml(actual?formatNumber(r.used,0):"N/A")} <span>/ ${escapeHtml(formatNumber(r?.available,0))}</span></div>
        <div class="fpga-resource-track"><div class="fpga-resource-fill ${actual?"":"unknown"}" style="width:${width}%"></div></div>
    </div>`;
}

function renderFpgaResources() {
    const d=state.fpgaResources; if(!d)return;
    $("fpgaDeviceName").textContent=`${d.board||"PYNQ-Z2"} / ${d.device||"XC7Z020"}`;
    $("resourceReportBadge").textContent=d.actual_usage_available?"Actual":"Needs Impl Report";
    $("resourceReportBadge").classList.toggle("actual",Boolean(d.actual_usage_available));
    const r=d.resources||{};
    $("fpgaResourceGrid").innerHTML=[resourceCard("LUT",r.lut),resourceCard("BRAM",r.bram),resourceCard("DSP",r.dsp),resourceCard("FF",r.ff)].join("");
    $("fpgaResourceNote").textContent=d.reason||"FPGA utilization report loaded.";
}

async function loadFpgaResources() {
    try { state.fpgaResources=await api("/api/fpga/resources"); renderFpgaResources(); }
    catch(error) { console.error("FPGA resources:",error); }
}

async function runSelectedInference() {
    if (state.inferenceMode === "fpga") await sendInformation();
    else await runLocalInference();
}

async function runLocalInference() {
    if (!state.segment || !state.segment.token) {
        $("inferenceBadge").textContent = "Error";

        setBox(
            $("inferenceState"),
            "Heartbeat chưa được load đúng. Hãy chọn lại Heartbeat / Segment.",
            "error"
        );

        return;
    }
    const button = $("runInferenceButton");
    button.disabled = true;
    $("inferenceBadge").textContent = "Running";
    setBox($("inferenceState"), "Running Local Integer mixed-precision model...");
    try {
        const result = await api("/api/infer/local", {
            method: "POST",
            headers: { "Content-Type": "application/json" },
            body: JSON.stringify({
                beat_token: state.segment.token,
            }),
        });
        $("predictionBox").classList.remove("hidden");
        $("predictedLabel").textContent = result.predicted_label;
        $("expectedLabel").textContent = result.true_label;
        $("statPredicted").textContent = result.predicted_label;
        $("statPredictedSub").textContent = result.correct ? "Correct" : "Different";
        $("statTrue").textContent = result.true_label;
        $("statTrueSub").textContent = result.record_id || state.segment.filename;
        $("inferenceBadge").textContent = "Completed";
        setBox($("inferenceState"), `Local Integer completed · total ${formatLatency(result.timing?.total_local_ms)}`, result.correct ? "success" : "error");
        updateExecutionSummary(result, "local");
        renderProbabilities(result);
        renderTrace(result.trace);
    } catch (error) {
        $("inferenceBadge").textContent = "Error";
        setBox($("inferenceState"), error.message, "error");
    } finally {
        button.disabled = false;
    }
}

function renderProbabilities(result) {
    const labels = ["N", "L", "R", "V", "A"];
    const colors = { N: "#52d273", L: "#5b95ff", R: "#ffb646", V: "#ff5d5d", A: "#b56cff" };
    const container = $("probabilityBars");
    container.innerHTML = "";
    labels.forEach((label, i) => {
        const p = Number(result.probabilities[i]) * 100;
        const row = document.createElement("div");
        row.className = "prob-row";
        row.innerHTML = `
            <div class="prob-head"><strong>${label}</strong><span>${p.toFixed(2)}%</span></div>
            <div class="prob-track"><div class="prob-fill" style="width:${Math.max(0, Math.min(100, p))}%; background: linear-gradient(90deg, ${colors[label]}, ${colors[label]});"></div></div>`;
        container.appendChild(row);
    });
}

function renderTrace(trace) {
    const container = $("traceBox");
    container.innerHTML = "";
    for (const [name, shape] of Object.entries(trace)) {
        const chip = document.createElement("span");
        chip.className = "trace-chip";
        chip.textContent = `${name}: ${shape.join("×")}`;
        container.appendChild(chip);
    }
}


function formatDuration(seconds) {
    if (seconds === null || seconds === undefined) return "-";

    const value = Math.max(0, Number(seconds));

    if (!Number.isFinite(value)) return "-";

    if (value < 60) {
        return `${value.toFixed(1)} s`;
    }

    const hours = Math.floor(value / 3600);
    const minutes = Math.floor((value % 3600) / 60);
    const secs = Math.floor(value % 60);

    if (hours > 0) {
        return `${hours}h ${minutes}m ${secs}s`;
    }

    return `${minutes}m ${secs}s`;
}

function renderFullTestConfusion(matrix) {
    const labels = ["N", "L", "R", "V", "A"];
    const table = $("fullTestConfusionTable");

    if (!table) return;

    const rows = Array.isArray(matrix) ? matrix : [];

    let html = `
        <thead>
            <tr>
                <th>True \\ Pred</th>
                ${labels.map(label => `<th>${label}</th>`).join("")}
            </tr>
        </thead>
        <tbody>
    `;

    labels.forEach((label, rowIndex) => {
        const row = Array.isArray(rows[rowIndex]) ? rows[rowIndex] : [];

        html += `
            <tr>
                <th>${label}</th>
                ${labels.map((_, colIndex) => {
                    const value = Number(row[colIndex] || 0);
                    const diagonal = rowIndex === colIndex ? " correct-cell" : "";
                    return `<td class="${diagonal.trim()}">${formatNumber(value, 0)}</td>`;
                }).join("")}
            </tr>
        `;
    });

    html += "</tbody>";

    table.innerHTML = html;
}

function renderFullTestJob(job) {
    if (!job) return;

    const status = String(job.status || "idle");
    const processed = Number(job.processed || 0);
    const total = Number(job.total || 0);
    const progress = Number(job.progress_percent || 0);
    const metrics = job.metrics || {};

    $("fullTestBadge").textContent =
        status.charAt(0).toUpperCase()
        + status.slice(1);

    $("fullTestBadge").classList.toggle(
        "running",
        ["preparing", "running", "cancelling"].includes(status)
    );

    $("fullTestBadge").classList.toggle(
        "completed",
        status === "completed"
    );

    $("fullTestBadge").classList.toggle(
        "failed",
        ["failed", "cancelled"].includes(status)
    );

    $("fullTestProgressWrap").classList.remove("hidden");
    $("fullTestMetrics").classList.remove("hidden");
    $("fullTestConfusionWrap").classList.remove("hidden");

    $("fullTestProgressText").textContent =
        `${formatNumber(processed, 0)} / ${formatNumber(total, 0)} samples`;

    $("fullTestProgressPercent").textContent =
        `${progress.toFixed(2)}%`;

    $("fullTestProgressBar").style.width =
        `${Math.max(0, Math.min(100, progress))}%`;

    $("fullTestAccuracy").textContent =
        Number.isFinite(Number(metrics.accuracy))
            ? `${(Number(metrics.accuracy) * 100).toFixed(3)}%`
            : "-";

    $("fullTestMacroF1").textContent =
        Number.isFinite(Number(metrics.macro_f1))
            ? Number(metrics.macro_f1).toFixed(6)
            : "-";

    $("fullTestAvgLatency").textContent =
        job.avg_total_latency_ms !== null
        && job.avg_total_latency_ms !== undefined
            ? formatLatency(job.avg_total_latency_ms)
            : "-";

    $("fullTestAvgFpga").textContent =
        job.mode === "fpga"
            ? (
                job.avg_fpga_weighted_ms !== null
                && job.avg_fpga_weighted_ms !== undefined
                    ? formatLatency(job.avg_fpga_weighted_ms)
                    : "-"
            )
            : "Not used";

    $("fullTestFailed").textContent =
        formatNumber(job.failed || 0, 0);

    const active = [
        "preparing",
        "running",
        "cancelling"
    ].includes(status);

    $("fullTestEta").textContent =
        active
            ? `ETA ${formatDuration(job.eta_seconds)}`
            : `Elapsed ${formatDuration(job.elapsed_seconds)}`;

    renderFullTestConfusion(
        job.confusion_matrix
    );

    const runButton = $("runFullTestButton");
    const stopButton = $("stopFullTestButton");

    runButton.disabled = active;
    stopButton.classList.toggle("hidden", !active);

    if (status === "completed") {
        setBox(
            $("fullTestState"),
            `Full test completed · ${formatNumber(metrics.evaluated || 0, 0)} evaluated · Accuracy ${(Number(metrics.accuracy || 0) * 100).toFixed(3)}% · Macro F1 ${Number(metrics.macro_f1 || 0).toFixed(6)}`,
            "success"
        );
    }
    else if (status === "cancelled") {
        setBox(
            $("fullTestState"),
            `Full test stopped at ${formatNumber(processed, 0)} / ${formatNumber(total, 0)} samples.`,
            "warn"
        );
    }
    else if (status === "failed") {
        setBox(
            $("fullTestState"),
            job.last_error || "Full test failed.",
            "error"
        );
    }
    else {
        const modeText =
            job.mode === "fpga"
                ? "PYNQ FPGA Hybrid"
                : "Local Integer";

        const current =
            job.current_true_label
                ? ` · current true=${job.current_true_label}`
                : "";

        setBox(
            $("fullTestState"),
            `Running ${modeText} · ${formatNumber(processed, 0)} / ${formatNumber(total, 0)}${current}`
        );
    }

    if (job.output_csv || job.summary_json) {
        $("fullTestOutputDetails").classList.remove("hidden");

        $("fullTestOutputPaths").textContent =
            [
                job.output_csv
                    ? `CSV: ${job.output_csv}`
                    : null,
                job.summary_json
                    ? `Summary: ${job.summary_json}`
                    : null,
            ]
            .filter(Boolean)
            .join("\n");
    }
}

function stopFullTestPolling() {
    if (state.fullTestPollTimer) {
        clearInterval(
            state.fullTestPollTimer
        );

        state.fullTestPollTimer = null;
    }
}

async function pollFullTestJob() {
    if (!state.fullTestJobId) return;

    try {
        const job = await api(
            `/api/test/jobs/${encodeURIComponent(state.fullTestJobId)}`
        );

        renderFullTestJob(job);

        if (
            ["completed", "failed", "cancelled"]
            .includes(job.status)
        ) {
            stopFullTestPolling();
        }
    }
    catch (error) {
        stopFullTestPolling();

        setBox(
            $("fullTestState"),
            error.message,
            "error"
        );

        $("runFullTestButton").disabled = false;
        $("stopFullTestButton").classList.add("hidden");
    }
}

function startFullTestPolling() {
    stopFullTestPolling();

    state.fullTestPollTimer = setInterval(
        pollFullTestJob,
        1500
    );
}

async function runFullTestSet() {
    const mode =
        state.inferenceMode === "fpga"
            ? "fpga"
            : "local";

    $("runFullTestButton").disabled = true;
    $("fullTestBadge").textContent = "Preparing";

    setBox(
        $("fullTestState"),
        mode === "fpga"
            ? "Preparing full test on PYNQ FPGA Hybrid. Large test sets can take a long time."
            : "Preparing full test on Local Integer model..."
    );

    try {
        const job = await api(
            "/api/test/run",
            {
                method: "POST",

                headers: {
                    "Content-Type":
                        "application/json"
                },

                body: JSON.stringify({
                    mode
                }),
            }
        );

        state.fullTestJobId =
            job.job_id;

        renderFullTestJob(job);
        startFullTestPolling();

        await pollFullTestJob();
    }
    catch (error) {
        $("runFullTestButton").disabled = false;
        $("fullTestBadge").textContent = "Error";

        setBox(
            $("fullTestState"),
            error.message,
            "error"
        );
    }
}

async function cancelFullTestSet() {
    if (!state.fullTestJobId) return;

    $("stopFullTestButton").disabled = true;

    try {
        const job = await api(
            `/api/test/jobs/${encodeURIComponent(state.fullTestJobId)}/cancel`,
            {
                method: "POST",
            }
        );

        renderFullTestJob(job);
    }
    catch (error) {
        setBox(
            $("fullTestState"),
            error.message,
            "error"
        );
    }
    finally {
        $("stopFullTestButton").disabled = false;
    }
}

async function restoreLatestFullTest() {
    try {
        const data = await api(
            "/api/test/latest"
        );

        const job = data?.job;

        if (!job) return;

        state.fullTestJobId =
            job.job_id;

        renderFullTestJob(job);

        if (
            ["preparing", "running", "cancelling"]
            .includes(job.status)
        ) {
            startFullTestPolling();
        }
    }
    catch (error) {
        console.error(
            "Full test restore:",
            error
        );
    }
}


async function sendInformation() {
    if (!state.segment) {
        return;
    }

    const button = $("runInferenceButton");
    button.disabled = true;
    $("inferenceBadge").textContent = "Running";
    setBox($("inferenceState"), "Running selected heartbeat on PYNQ-Z2 FPGA...");

    try {
        const result = await api(
            "/api/send",
            {
                method: "POST",

                headers: {
                    "Content-Type":
                        "application/json"
                },

                body: JSON.stringify({
                    beat_token:
                        state.segment.token,
                }),
            }
        );

        $("sendResponse").textContent =
            JSON.stringify(
                result,
                null,
                2
            );

        if (
            result.success
            && result.response
            && result.response.predicted_label
        ) {
            const fpga =
                result.response;

            $("predictionBox")
                .classList
                .remove("hidden");

            $("predictedLabel").textContent =
                fpga.predicted_label;

            $("expectedLabel").textContent =
                fpga.true_label
                || state.segment.label;

            $("statPredicted").textContent = fpga.predicted_label;
            $("statPredictedSub").textContent = fpga.correct ? "Correct · PYNQ FPGA" : "Different · PYNQ FPGA";
            $("statTrue").textContent = fpga.true_label || state.segment.label;
            $("statTrueSub").textContent = state.segment.filename || state.segment.record_id || "-";
            $("inferenceBadge").textContent = "Completed";
            setBox($("inferenceState"), `PYNQ FPGA Hybrid completed · FPGA weighted ${formatLatency(fpga.timing?.fpga_weighted_ms)} · total ${formatLatency(fpga.timing?.total_hybrid_ms)}`, fpga.correct ? "success" : "error");
            updateExecutionSummary(fpga, "fpga");
            renderProbabilities(fpga);
            renderTrace(fpga.trace || {});
            $("fpgaResourcePanel").classList.remove("hidden");
            $("pynqDebugDetails").classList.remove("hidden");
        }
        else {
            $("inferenceBadge").textContent = "Error";
            setBox($("inferenceState"), result.message || result.error || "PYNQ request failed.", "error");
            $("pynqDebugDetails").classList.remove("hidden");
        }
    }
    catch (error) {
        $("sendResponse").textContent =
            error.message;

        $("inferenceBadge").textContent = "Error";
        setBox($("inferenceState"), error.message, "error");
        $("pynqDebugDetails").classList.remove("hidden");
    }
    finally {
        button.disabled = false;
    }
}


async function loadCurrentArchitecture() {
    const nas = await api("/api/nas/current");
    state.currentNas = nas;
    $("layerCount").textContent = nas.weighted_layers;
    $("int8Count").textContent = nas.int8_count;
    $("int4Count").textContent = nas.int4_count;
    $("deploymentName").textContent = nas.deployment;
    $("deploymentNameCenter").textContent = nas.deployment;
    $("int8CountCenter").textContent = nas.int8_count;
    $("int4CountCenter").textContent = nas.int4_count;
    const total = Math.max(1, nas.int8_count + nas.int4_count);
    $("int8Percent").textContent = `${Math.round(100 * nas.int8_count / total)}%`;
    $("int4Percent").textContent = `${Math.round(100 * nas.int4_count / total)}%`;
    renderArchitecture(nas.architecture);
    renderPrecisionTable(nas);
}

function precisionBadge(bits) {
    const cls = Number(bits) === 4 ? "int4" : "int8";
    return `<span class="badge ${cls}">INT${bits}</span>`;
}

function renderArchitecture(arch) {
    const container = $("architectureFlow");
    container.innerHTML = "";
    const stem = document.createElement("div");
    stem.className = "arch-card";
    stem.innerHTML = `<h3>Input ${formatShape(arch.input_shape)} → Stem · K${arch.stem.kernel ?? "-"} · C${arch.stem.output_channels ?? "-"}</h3><div class="branch-sub">${precisionBadge(arch.stem.bits)}</div>`;
    container.appendChild(stem);
    for (const block of arch.blocks) {
        const card = document.createElement("div");
        card.className = "arch-card";
        const branches = [...block.branches];
        if (block.pool_branch) branches.push({ ...block.pool_branch, display_name: "Pool" });
        card.innerHTML = `
            <h3>${escapeHtml(block.name)} · out C${block.output_channels ?? "-"}${block.downsample ? " · ↓2" : ""}</h3>
            <div class="arch-branch-grid">
                ${branches.map((branch, index) => `
                    <div class="branch-card">
                        <div class="branch-name">${escapeHtml(branch.display_name || `B${index + 1}`)}</div>
                        <div class="branch-main">K${branch.kernel ?? "-"} · C${branch.output_channels ?? "-"}</div>
                        <div class="branch-sub">${precisionBadge(branch.bits)} · d${branch.dilation ?? 1}</div>
                    </div>`).join("")}
            </div>`;
        container.appendChild(card);
    }
}

function renderPrecisionTable(nas) {
    const chips = $("int4LayerChips");
    chips.innerHTML = "";
    for (const layer of nas.int4_layers) {
        const chip = document.createElement("span");
        chip.className = "layer-chip int4";
        chip.textContent = layer;
        chips.appendChild(chip);
    }
    const tbody = $("layerTableBody");
    tbody.innerHTML = "";
    for (const layer of nas.layers) {
        const row = document.createElement("tr");
        row.innerHTML = `<td title="${escapeHtml(layer.name)}">${escapeHtml(layer.name)}</td><td>${precisionBadge(layer.bits)}</td><td>${escapeHtml(formatShape(layer.kernel_shape))}</td>`;
        tbody.appendChild(row);
    }
}

function checkedValues(selector) {
    return Array.from(document.querySelectorAll(selector)).filter(x => x.checked).map(x => Number(x.value));
}

function parseIntegerList(text) {
    return String(text).split(",").map(x => Number(x.trim())).filter(x => Number.isInteger(x) && x > 0);
}

function optionalNumber(id) {
    const text = $(id).value.trim();
    if (text === "") return null;
    const value = Number(text);
    return Number.isFinite(value) ? value : null;
}



function setNasActivityState(
    status,
    progress = null
) {
    const card =
        $("nasActivityCard");

    card.classList.remove(
        "idle",
        "checking",
        "running",
        "completed",
        "failed"
    );

    const normalizedStatus =
        String(
            status || "idle"
        ).toLowerCase();

    let className = "idle";
    let title = "NAS Idle";
    let subtitle =
        "Waiting for architecture search.";
    let statusText = "Idle";

    if (
        normalizedStatus === "checking"
        || normalizedStatus === "loading_train_val"
        || normalizedStatus === "importing_project_modules"
        || normalizedStatus === "loading_dataset"
    ) {
        className = "checking";
        title = "Preparing NAS";
        subtitle =
            "Checking environment and loading training/validation data.";
        statusText = "Preparing";
    }
    else if (
        normalizedStatus === "running"
        || normalizedStatus === "training"
        || normalizedStatus === "training_candidate"
    ) {
        className = "running";
        title = "NAS Search Running";
        subtitle =
            "A candidate architecture is being trained and evaluated.";
        statusText = "Running";
    }
    else if (
        normalizedStatus === "completed"
    ) {
        className = "completed";
        title = "NAS Search Completed";
        subtitle =
            "Best architecture selected from completed candidates.";
        statusText = "Completed";
    }
    else if (
        normalizedStatus === "failed"
        || normalizedStatus === "waiting_engine"
        || normalizedStatus === "error"
    ) {
        className = "failed";
        title = "NAS Search Failed";
        subtitle =
            "Search stopped. Open NAS job details for the error.";
        statusText = "Failed";
    }

    card.classList.add(
        className
    );

    $("nasActivityTitle").textContent =
        title;

    $("nasActivitySubtitle").textContent =
        subtitle;

    $("nasActivityStatus").textContent =
        statusText;

    let completed = 0;
    let total = 0;
    let candidate = "-";
    let percent = 0;

    if (progress) {
        completed =
            Number(
                progress.completed
            ) || 0;

        total =
            Number(
                progress.total
            ) || 0;

        if (
            progress.candidate_id
            !== undefined
            && progress.candidate_id
            !== null
        ) {
            candidate =
                `#${Number(progress.candidate_id) + 1}`;
        }

        if (total > 0) {
            percent =
                Math.max(
                    0,
                    Math.min(
                        100,
                        Math.round(
                            (
                                completed
                                / total
                            )
                            * 100
                        )
                    )
                );
        }

        const arch =
            progress.architecture;

        if (arch) {
            const blocks = [
                ["B1", arch.block1_kernels, arch.block1_channels],
                ["B2", arch.block2_kernels, arch.block2_channels],
                ["B3", arch.block3_kernels, arch.block3_channels],
                ["B4", arch.block4_kernels, arch.block4_channels],
            ];

            $("nasCurrentArchitecture").innerHTML =
                blocks
                    .map(
                        ([name, kernels, channels]) => {
                            const kernelText =
                                Array.isArray(kernels)
                                    ? kernels
                                        .map(k => `K${k}`)
                                        .join("/")
                                    : "-";

                            return `
                                <span class="nas-arch-chip">
                                    ${name}: ${kernelText} · C${channels ?? "-"}
                                </span>
                            `;
                        }
                    )
                    .join("");
        }
    }

    if (
        normalizedStatus === "completed"
    ) {
        percent = 100;
    }

    $("nasCurrentCandidate").textContent =
        candidate;

    $("nasCompletedCandidates").textContent =
        `${completed} / ${total}`;

    $("nasActivityPercent").textContent =
        `${percent}%`;

    $("nasProgressBar").style.width =
        `${percent}%`;

    if (
        className !== "running"
        && className !== "checking"
        && !progress?.architecture
    ) {
        $("nasCurrentArchitecture").textContent =
            className === "completed"
                ? "Search finished. See Candidate History for the best model."
                : (
                    className === "failed"
                        ? "No candidate is currently running."
                        : "No candidate is being evaluated."
                );
    }
}


function resetNasActivity() {
    setNasActivityState(
        "idle",
        {
            completed: 0,
            total: 0,
        }
    );
}


async function startNasSearch() {
    const payload = {
        kernels:
            checkedValues(
                ".kernel-option"
            ),

        channels:
            parseIntegerList(
                $("nasChannels").value
            ),

        trials:
            Number(
                $("nasTrials").value
            ),

        objective:
            $("nasObjective").value,

        epochs_per_candidate:
            Number(
                $("nasEpochs").value
            ),
    };

    if (payload.kernels.length < 3) {
        setBox(
            $("nasState"),
            "Chọn ít nhất 3 kernel vì mỗi Inception block có 3 kernel branches.",
            "error"
        );
        return;
    }

    if (payload.channels.length === 0) {
        setBox(
            $("nasState"),
            "Channel Candidates đang rỗng.",
            "error"
        );
        return;
    }

    if (
        !Number.isFinite(payload.trials)
        || payload.trials < 1
    ) {
        setBox(
            $("nasState"),
            "Trials phải >= 1.",
            "error"
        );
        return;
    }

    if (
        !Number.isFinite(
            payload.epochs_per_candidate
        )
        || payload.epochs_per_candidate < 1
    ) {
        setBox(
            $("nasState"),
            "Epochs / Candidate phải >= 1.",
            "error"
        );
        return;
    }

    const button =
        $("startNasButton");

    button.disabled = true;

    try {
        setBox(
            $("nasState"),
            "Checking NAS environment..."
        );

        setNasActivityState(
            "checking",
            {
                completed: 0,
                total: payload.trials,
            }
        );

        const preflight =
            await api(
                "/api/nas/preflight"
            );

        $("nasJobJson").textContent =
            JSON.stringify(
                preflight,
                null,
                2
            );

        if (!preflight.ok) {
            setBox(
                $("nasState"),
                "NAS preflight failed. Open NAS job details to see missing source/data/packages.",
                "error"
            );

            setNasActivityState(
                "failed",
                {
                    completed: 0,
                    total: payload.trials,
                }
            );

            return;
        }

        setBox(
            $("nasState"),
            "Starting architecture search..."
        );

        setNasActivityState(
            "running",
            {
                completed: 0,
                total: payload.trials,
            }
        );

        const result =
            await api(
                "/api/nas/search",
                {
                    method: "POST",

                    headers: {
                        "Content-Type":
                            "application/json"
                    },

                    body:
                        JSON.stringify(
                            payload
                        ),
                }
            );

        $("nasJobJson").textContent =
            JSON.stringify(
                result,
                null,
                2
            );

        const mode =
            result.status === "running"
                ? "success"
                : (
                    result.status
                    === "waiting_engine"
                        ? "warn"
                        : ""
                );

        setBox(
            $("nasState"),
            `${result.status}: ${result.message || ""}`,
            mode
        );

        if (
            result.job_id
            && result.status === "running"
        ) {
            pollNasJob(
                result.job_id
            );
        }
    }
    catch (error) {
        setBox(
            $("nasState"),
            error.message,
            "error"
        );

        setNasActivityState(
            "failed"
        );
    }
    finally {
        button.disabled = false;
    }
}



function pollNasJob(jobId) {
    if (state.nasPollTimer) {
        clearInterval(
            state.nasPollTimer
        );
    }

    const pollOnce =
        async () => {
            try {
                const job =
                    await api(
                        `/api/nas/jobs/${encodeURIComponent(jobId)}`
                    );

                $("nasJobJson").textContent =
                    JSON.stringify(
                        job,
                        null,
                        2
                    );

                const progress =
                    job.progress || null;

                // progress.json has more precise states:
                // importing_project_modules, loading_train_val,
                // training_candidate, completed...
                const visualStatus =
                    progress?.status
                    || job.status;

                setNasActivityState(
                    visualStatus,
                    progress
                );

                let nasMessage =
                    `${job.status}: ${job.message || ""}`;

                if (
                    progress
                    && progress.status === "training_candidate"
                ) {
                    const current =
                        Number(
                            progress.candidate_id
                        ) + 1;

                    nasMessage =
                        `Running candidate ${current}/${progress.total}`;
                }

                if (
                    job.status === "failed"
                    && job.stderr_tail
                ) {
                    const stderrLines =
                        String(
                            job.stderr_tail
                        )
                        .trim()
                        .split("\n");

                    nasMessage +=
                        `\n${stderrLines.slice(-8).join("\n")}`;
                }

                setBox(
                    $("nasState"),
                    nasMessage,
                    job.status === "completed"
                        ? "success"
                        : (
                            job.status === "failed"
                                ? "error"
                                : ""
                        )
                );

                if (
                    job.status === "completed"
                    || job.status === "failed"
                ) {
                    if (
                        state.nasPollTimer
                    ) {
                        clearInterval(
                            state.nasPollTimer
                        );

                        state.nasPollTimer =
                            null;
                    }

                    if (
                        job.status
                        === "completed"
                    ) {
                        setNasActivityState(
                            "completed",
                            progress || {
                                completed: 1,
                                total: 1,
                            }
                        );
                    }
                    else {
                        setNasActivityState(
                            "failed",
                            progress
                        );
                    }

                    loadCandidateJobs(
                        false
                    ).then(
                        () =>
                            loadCandidates(
                                state.candidateJobId
                            )
                    );
                }
            }
            catch (error) {
                console.error(
                    error
                );

                setNasActivityState(
                    "failed"
                );
            }
        };

    // Don't wait two seconds before showing the first server-side state.
    pollOnce();

    state.nasPollTimer =
        setInterval(
            pollOnce,
            1500
        );
}



async function loadCandidateJobs(
    preserveJob = true
) {
    const data =
        await api(
            "/api/candidate-jobs"
        );

    state.candidateJobs =
        data.jobs || [];

    const select =
        $("candidateJobSelect");

    const previous =
        preserveJob
            ? state.candidateJobId
            : null;

    select.innerHTML = "";

    if (
        state.candidateJobs.length === 0
    ) {
        const option =
            document.createElement(
                "option"
            );

        option.value = "";
        option.textContent =
            "No NAS search run";

        select.appendChild(
            option
        );

        state.candidateJobId =
            null;

        return null;
    }

    for (
        const job
        of state.candidateJobs
    ) {
        const option =
            document.createElement(
                "option"
            );

        option.value =
            job.job_id;

        option.textContent =
            `${job.job_id}`
            + ` · ${job.candidate_count} candidates`
            + (
                job.status
                    ? ` · ${job.status}`
                    : ""
            );

        select.appendChild(
            option
        );
    }

    const exists =
        previous
        && state.candidateJobs.some(
            job =>
                job.job_id
                === previous
        );

    state.candidateJobId =
        exists
            ? previous
            : state.candidateJobs[0]
                .job_id;

    select.value =
        state.candidateJobId;

    return state.candidateJobId;
}


async function loadCandidates(
    jobId = null
) {
    try {
        if (!jobId) {
            jobId =
                await loadCandidateJobs(
                    true
                );
        }

        if (!jobId) {
            state.candidates = [];
            state.activeCandidateIndex = -1;

            renderCandidateTable(
                []
            );

            renderCandidateSelector(
                []
            );

            $("candidateSummaryCard")
                .classList
                .add("hidden");

            setBox(
                $("candidateNotice"),
                "No NAS candidate history found yet.",
                "warn"
            );

            return;
        }

        state.candidateJobId =
            jobId;

        const data = await api(
            `/api/candidates?job_id=${encodeURIComponent(jobId)}`
        );

        state.candidates =
            data.candidates || [];

        state.selectedCandidateId =
            data.selected_candidate_id;

        state.bestCandidateId =
            data.best_candidate_id;

        if (
            state.candidates.length === 0
        ) {
            setBox(
                $("candidateNotice"),
                "This NAS run has no candidate results.",
                "warn"
            );
        }
        else {
            const selectedText =
                state.selectedCandidateId
                !== null
                && state.selectedCandidateId
                !== undefined
                    ? ` · selected #${state.selectedCandidateId}`
                    : "";

            setBox(
                $("candidateNotice"),
                `${state.candidates.length} candidates loaded`
                + selectedText
                + ".",
                "success"
            );
        }

        renderCandidateTable(
            state.candidates
        );

        renderCandidateSelector(
            state.candidates
        );

        let index =
            state.candidates.findIndex(
                candidate =>
                    candidate.is_selected
            );

        if (index < 0) {
            index =
                state.candidates.findIndex(
                    candidate =>
                        candidate.is_best
                );
        }

        if (
            index < 0
            && state.candidates.length > 0
        ) {
            index = 0;
        }

        if (index >= 0) {
            showCandidate(
                index
            );
        }
        else {
            $("candidateSummaryCard")
                .classList
                .add("hidden");
        }
    }
    catch (error) {
        setBox(
            $("candidateNotice"),
            error.message,
            "error"
        );
    }
}


function renderCandidateSelector(
    candidates
) {
    const select =
        $("candidateSelect");

    select.innerHTML = "";

    if (!candidates.length) {
        const option =
            document.createElement(
                "option"
            );

        option.value = "";
        option.textContent =
            "No candidate";

        select.appendChild(
            option
        );

        return;
    }

    candidates.forEach(
        (candidate, index) => {
            const option =
                document.createElement(
                    "option"
                );

            option.value =
                String(index);

            const badges = [
                candidate.is_best
                    ? "BEST"
                    : null,

                candidate.is_selected
                    ? "SELECTED"
                    : null,
            ]
                .filter(Boolean)
                .join(", ");

            option.textContent =
                `Candidate #${candidate.id}`
                + (
                    candidate.macro_f1
                    !== null
                    && candidate.macro_f1
                    !== undefined

                        ? ` · F1 ${formatNumber(candidate.macro_f1, 4)}`
                        : ""
                )
                + (
                    badges
                        ? ` · ${badges}`
                        : ""
                );

            select.appendChild(
                option
            );
        }
    );
}


function architectureBlock(
    blockName,
    kernels,
    channels
) {
    const kernelList =
        Array.isArray(kernels)
            ? kernels
                .map(
                    kernel =>
                        `K${kernel}`
                )
                .join(" · ")
            : "-";

    return `
        <div class="candidate-block-card">
            <div class="candidate-block-head">
                <strong>${escapeHtml(blockName)}</strong>
                <span class="changed-badge">NAS CHANGED</span>
            </div>

            <div class="candidate-block-row">
                <span>Branch kernels</span>
                <strong>${escapeHtml(kernelList)}</strong>
            </div>

            <div class="candidate-block-row">
                <span>Output channels</span>
                <strong>C${escapeHtml(channels ?? "-")}</strong>
            </div>
        </div>
    `;
}


function showCandidate(index) {
    const candidate =
        state.candidates[index];

    if (!candidate) {
        return;
    }

    state.activeCandidateIndex =
        index;

    $("candidateSelect").value =
        String(index);

    $("candidateSummaryCard")
        .classList
        .remove("hidden");

    $("candidateDisplayName").textContent =
        `Candidate #${candidate.id}`;

    $("candidateStatusText").textContent =
        `Status: ${candidate.status || "-"}`;

    $("candidateF1").textContent =
        formatNumber(
            candidate.macro_f1,
            4
        );

    $("candidateAccuracy").textContent =
        candidate.accuracy === null
        || candidate.accuracy === undefined
            ? "-"
            : `${(
                Number(
                    candidate.accuracy
                )
                * 100
            ).toFixed(2)}%`;

    $("candidateParams").textContent =
        formatNumber(
            candidate.params,
            0
        );

    $("candidateMacs").textContent =
        formatNumber(
            candidate.macs,
            0
        );

    $("candidateCheckpointStatus").textContent =
        candidate.checkpoint_available
            ? "SAVED"
            : "NOT SAVED";

    $("candidateEpochsRan").textContent =
        candidate.epochs_ran === null
        || candidate.epochs_ran === undefined
            ? "-"
            : formatNumber(
                candidate.epochs_ran,
                0
            );

    $("candidateCheckpointPath").textContent =
        candidate.checkpoint_available
            ? `Checkpoint: ${candidate.checkpoint_path || "-"}`
            : "Checkpoint: not available for this candidate (old NAS runs will not have one).";

    $("candidateBestBadge")
        .classList
        .toggle(
            "hidden",
            !candidate.is_best
        );

    $("candidateSelectedBadge")
        .classList
        .toggle(
            "hidden",
            !candidate.is_selected
        );

    const arch =
        candidate.architecture
        || {};

    $("candidateArchitectureGrid")
        .innerHTML =
            [
                architectureBlock(
                    "Block 1",
                    arch.block1_kernels,
                    arch.block1_channels
                ),

                architectureBlock(
                    "Block 2",
                    arch.block2_kernels,
                    arch.block2_channels
                ),

                architectureBlock(
                    "Block 3",
                    arch.block3_kernels,
                    arch.block3_channels
                ),

                architectureBlock(
                    "Block 4",
                    arch.block4_kernels,
                    arch.block4_channels
                ),
            ].join("");

    const button =
        $("selectCandidateButton");

    const selectable =
        candidate.status === "completed"
        || candidate.status === "feasible"
        || candidate.status === "unknown";

    button.disabled =
        !selectable;

    if (candidate.is_selected) {
        button.textContent =
            "✓ Selected for Next Stage";
    }
    else {
        button.textContent =
            "Select Candidate for Next Stage";
    }

    document
        .querySelectorAll(
            ".candidate-row"
        )
        .forEach(
            row =>
                row.classList.remove(
                    "active"
                )
        );

    const activeRow =
        document.querySelector(
            `.candidate-row[data-index="${index}"]`
        );

    if (activeRow) {
        activeRow.classList.add(
            "active"
        );
    }
}


function renderCandidateTable(
    candidates
) {
    const tbody =
        $("candidateTableBody");

    tbody.innerHTML = "";

    candidates.forEach(
        (candidate, index) => {
            const row =
                document.createElement(
                    "tr"
                );

            row.dataset.index =
                String(index);

            row.className =
                "candidate-row"
                + (
                    candidate.is_best
                        ? " best"
                        : ""
                )
                + (
                    candidate.is_selected
                        ? " selected"
                        : ""
                );

            const title =
                `#${candidate.id}`
                + (
                    candidate.is_best
                        ? " ★"
                        : ""
                )
                + (
                    candidate.is_selected
                        ? " ✓"
                        : ""
                );

            row.innerHTML =
                `<td>${escapeHtml(title)}</td>`
                + `<td>${formatNumber(candidate.macro_f1, 4)}</td>`
                + `<td>${formatNumber(candidate.params, 0)}</td>`
                + `<td>${formatNumber(candidate.macs, 0)}</td>`
                + `<td>${candidate.checkpoint_available ? "Saved" : "-"}</td>`;

            row.addEventListener(
                "click",
                () =>
                    showCandidate(
                        index
                    )
            );

            tbody.appendChild(
                row
            );
        }
    );
}


async function selectActiveCandidate() {
    const index =
        state.activeCandidateIndex;

    const candidate =
        state.candidates[index];

    if (!candidate) {
        return;
    }

    const button =
        $("selectCandidateButton");

    button.disabled = true;

    try {
        const result =
            await api(
                "/api/candidates/select",
                {
                    method:
                        "POST",

                    headers: {
                        "Content-Type":
                            "application/json"
                    },

                    body:
                        JSON.stringify(
                            {
                                candidate_id:
                                    candidate.id,

                                job_id:
                                    state.candidateJobId,
                            }
                        ),
                }
            );

        state.selectedCandidateId =
            result.candidate_id;

        setBox(
            $("candidateNotice"),
            `Candidate #${result.candidate_id} selected for the next stage.`,
            "success"
        );

        await loadCandidates(
            state.candidateJobId
        );
    }
    catch (error) {
        setBox(
            $("candidateNotice"),
            error.message,
            "error"
        );
    }
    finally {
        button.disabled = false;
    }
}


function setupEvents() {
    $("recordSelect")
        .addEventListener(
            "change",
            async () => {
                await selectRecord(
                    $("recordSelect").value
                );
            }
        );

    $("channelSelect")
        .addEventListener(
            "change",
            async () => {
                if (state.record) {
                    await loadRecordWaveform(
                        state.record.record_id,
                        $("channelSelect").value
                    );
                }
            }
        );

    $("segmentSelect")
        .addEventListener(
            "change",
            async () => {
                const token =
                    $("segmentSelect").value;

                if (token) {
                    await loadSegment(
                        token,
                        true
                    );
                }
            }
        );

    $("loadSegmentButton")
        .addEventListener(
            "click",
            async () => {
                const token =
                    $("segmentSelect").value;

                if (token) {
                    await loadSegment(
                        token,
                        true
                    );
                }
            }
        );

    $("classFilter")
        .addEventListener(
            "change",
            async () => {
                state.classFilter =
                    $("classFilter").value;

                await loadRecords();
            }
        );

    $("beatSource")
        .addEventListener(
            "change",
            async () => {
                state.beatSource =
                    $("beatSource").value;

                if (state.record) {
                    await loadSegments(
                        state.record.record_id
                    );
                }
            }
        );

    $("refreshRecordsButton")
        .addEventListener(
            "click",
            async () => {
                setBox(
                    $("inferenceState"),
                    "Rescanning datasets..."
                );

                await api(
                    "/api/dataset/refresh",
                    {
                        method: "POST"
                    }
                );

                await loadStatus();
                await loadRecords();

                setBox(
                    $("inferenceState"),
                    "Dataset reloaded.",
                    "success"
                );
            }
        );

    $("recordPositionSlider")
        .addEventListener(
            "input",
            () => {
                scheduleWindowLoad(
                    $("recordPositionSlider").value
                );
            }
        );

    $("windowSecondsSelect")
        .addEventListener(
            "change",
            async () => {
                const previousCenter =
                    state.windowStartSec
                    + state.windowSeconds / 2;

                state.windowSeconds =
                    Math.max(
                        1,
                        Number(
                            $("windowSecondsSelect").value
                        ) || 10
                    );

                configureRecordSlider();

                await loadRecordWindow(
                    Math.max(
                        0,
                        previousCenter
                        - state.windowSeconds / 2
                    )
                );
            }
        );

    $("previousWindowButton")
        .addEventListener(
            "click",
            async () => {
                await loadRecordWindow(
                    state.windowStartSec
                    - state.windowSeconds
                );
            }
        );

    $("nextWindowButton")
        .addEventListener(
            "click",
            async () => {
                await loadRecordWindow(
                    state.windowStartSec
                    + state.windowSeconds
                );
            }
        );

    $("rawButton")
        .addEventListener(
            "click",
            () => {
                state.segmentMode = "raw";

                $("rawButton")
                    .classList
                    .add("active");

                $("normalizedButton")
                    .classList
                    .remove("active");

                drawSegmentWaveform();
            }
        );

    $("normalizedButton")
        .addEventListener(
            "click",
            () => {
                state.segmentMode =
                    "normalized";

                $("normalizedButton")
                    .classList
                    .add("active");

                $("rawButton")
                    .classList
                    .remove("active");

                drawSegmentWaveform();
            }
        );

    $("modeLocalButton").addEventListener("click", () => setExecutionMode("local"));
    $("modeFpgaButton").addEventListener("click", () => setExecutionMode("fpga"));
    $("runInferenceButton").addEventListener("click", runSelectedInference);
    $("runFullTestButton").addEventListener("click", runFullTestSet);
    $("stopFullTestButton").addEventListener("click", cancelFullTestSet);

    $("startNasButton")
        .addEventListener(
            "click",
            startNasSearch
        );

    $("refreshCandidatesButton")
        .addEventListener(
            "click",
            async () => {
                await loadCandidateJobs(
                    true
                );

                await loadCandidates(
                    state.candidateJobId
                );
            }
        );

    $("candidateJobSelect")
        .addEventListener(
            "change",
            async () => {
                state.candidateJobId =
                    $("candidateJobSelect").value
                    || null;

                await loadCandidates(
                    state.candidateJobId
                );
            }
        );

    $("candidateSelect")
        .addEventListener(
            "change",
            () => {
                const index =
                    Number(
                        $("candidateSelect").value
                    );

                if (
                    Number.isInteger(index)
                    && index >= 0
                ) {
                    showCandidate(
                        index
                    );
                }
            }
        );

    $("selectCandidateButton")
        .addEventListener(
            "click",
            selectActiveCandidate
        );

    window.addEventListener(
        "resize",
        () => {
            drawWindowWaveform();
            drawOverviewWaveform();
            drawSegmentWaveform();
        }
    );
}



async function boot() {
    setupEvents();
    updateDateTime();
    resetNasActivity();
    updateExecutionModeUi();
    loadFpgaResources();

    setInterval(
        updateDateTime,
        1000
    );

    // Force ATR as the first usable source.
    state.beatSource = "atr";
    $("beatSource").value = "atr";

    $("datasetError").textContent = "";

    try {
        await loadStatus();

        // Load dataset first so selectors are usable even if
        // model/candidate endpoints fail later.
        await loadRecords();

        // Architecture/candidate failures should not block ECG selection.
        loadCurrentArchitecture()
            .catch(
                error => console.error(
                    "Architecture:",
                    error
                )
            );

        loadCandidates()
            .catch(
                error => console.error(
                    "Candidates:",
                    error
                )
            );
    }
    catch (error) {
        $("serverText").textContent =
            "Dataset initialization failed";

        $("datasetError").textContent =
            `ERROR: ${error.message}`;

        $("datasetStatus").textContent =
            "Cannot load records.";

        console.error(error);
    }
}


boot();

window.addEventListener("load", restoreLatestFullTest);
