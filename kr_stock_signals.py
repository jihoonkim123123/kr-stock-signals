# ---------------------------------------------------------------------------
# 5) 개선된 HTML 대시보드 UI
# ---------------------------------------------------------------------------

HTML = r"""
<!DOCTYPE html>
<html lang="ko">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>KR Stock Signals Dashboard</title>
    <script src="https://cdn.tailwindcss.com"></script>
    <link href="https://cdn.jsdelivr.net/npm/gridjs/dist/theme/mermaid.min.css" rel="stylesheet">
    <script src="https://cdn.jsdelivr.net/npm/gridjs/dist/gridjs.umd.js"></script>
    <style>
        @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&display=swap');
        body { font-family: 'Inter', -apple-system, "Apple SD Gothic Neo", sans-serif; background-color: #0f172a; color: #f1f5f9; margin: 0; }
        .gridjs-container { background: transparent; color: #f1f5f9; padding: 0; }
        .gridjs-table { background-color: #1e293b; border-radius: 0.75rem; overflow: hidden; border: none; width: 100%; }
        .gridjs-th { background-color: #334155 !important; color: #94a3b8 !important; font-weight: 600; text-transform: uppercase; font-size: 0.75rem; letter-spacing: 0.05em; padding: 1.25rem !important; border: none !important; }
        .gridjs-td { background-color: transparent !important; color: #e2e8f0 !important; border-bottom: 1px solid #334155 !important; padding: 1rem 1.25rem !important; }
        .gridjs-tr:hover .gridjs-td { background-color: #334155 !important; }
        .gridjs-search-input { background-color: #1e293b !important; border: 1px solid #475569 !important; color: #f1f5f9 !important; border-radius: 0.5rem !important; padding: 0.6rem 1rem !important; width: 300px !important; }
        .gridjs-pagination .gridjs-pages button { background-color: #334155 !important; color: #f1f5f9 !important; border: none !important; border-radius: 0.375rem !important; margin: 0 0.25rem !important; }
        .gridjs-pagination .gridjs-pages button.gridjs-currentPage { background-color: #3b82f6 !important; font-weight: bold !important; }
        .pill { display: inline-flex; align-items: center; padding: 0.25rem 0.75rem; border-radius: 9999px; font-size: 0.75rem; font-weight: 700; }
        .pill-buy { background-color: rgba(16, 185, 129, 0.15); color: #10b981; border: 1px solid rgba(16, 185, 129, 0.3); }
        .pill-neutral { background-color: rgba(245, 158, 11, 0.15); color: #f59e0b; border: 1px solid rgba(245, 158, 11, 0.3); }
        .pill-sell { background-color: rgba(100, 116, 139, 0.15); color: #94a3b8; border: 1px solid rgba(100, 116, 139, 0.3); }
    </style>
</head>
<body class="p-6 md:p-12">
    <div class="max-w-7xl mx-auto">
        <div class="flex flex-col md:flex-row md:items-center justify-between mb-10 gap-6">
            <div>
                <h1 class="text-3xl font-extrabold text-white tracking-tight flex items-center gap-3">
                    <span class="bg-blue-600 p-2 rounded-lg text-2xl">📈</span> KR Stock Signals
                </h1>
                <p class="text-slate-400 mt-2 font-medium">코스피 200 & 코스닥 150 정밀 매매 신호 분석</p>
            </div>
            <div id="regimeBadge"></div>
        </div>

        <div class="grid grid-cols-1 md:grid-cols-3 gap-6 mb-10">
            <div class="bg-slate-800/40 border border-slate-700/50 p-6 rounded-2xl backdrop-blur-sm shadow-xl">
                <p class="text-slate-500 text-xs font-bold uppercase tracking-widest mb-2">분석 기준일</p>
                <h2 class="text-2xl font-bold text-white">__DATE__</h2>
            </div>
            <div class="bg-slate-800/40 border border-slate-700/50 p-6 rounded-2xl backdrop-blur-sm shadow-xl">
                <p class="text-slate-500 text-xs font-bold uppercase tracking-widest mb-2">스캔 종목수</p>
                <h2 class="text-2xl font-bold text-white" id="totalCount">-</h2>
            </div>
            <div class="bg-slate-800/40 border border-slate-700/50 p-6 rounded-2xl backdrop-blur-sm shadow-xl border-l-4 border-l-emerald-500">
                <p class="text-slate-500 text-xs font-bold uppercase tracking-widest mb-2">진입 후보 (80점+)</p>
                <h2 class="text-2xl font-bold text-emerald-400" id="highScoreCount">-</h2>
            </div>
        </div>

        <div class="flex gap-2 mb-8 bg-slate-800/60 p-1.5 rounded-2xl w-fit border border-slate-700/50 shadow-inner">
            <button onclick="changeMode('trend')" id="btn-trend" class="tab-btn px-8 py-3 rounded-xl text-sm font-bold transition-all bg-blue-600 text-white shadow-lg shadow-blue-900/40">장기 추세 전략</button>
            <button onclick="changeMode('swing')" id="btn-swing" class="tab-btn px-8 py-3 rounded-xl text-sm font-bold transition-all text-slate-400 hover:text-white hover:bg-slate-700/50">단기 스윙 (참고)</button>
        </div>

        <div class="bg-slate-800/30 border border-slate-700/50 rounded-2xl shadow-2xl overflow-hidden backdrop-blur-md">
            <div id="tableWrapper"></div>
        </div>

        <footer class="mt-16 text-center border-t border-slate-800 pt-10 pb-10">
            <p class="text-slate-500 text-sm">© 2026 KR Stock Signals Algorithm. All Rights Reserved.</p>
            <div class="mt-4 flex justify-center gap-4 text-[11px] text-slate-600 font-medium">
                <span>데이터: KRX/FDR</span>
                <span>•</span>
                <span>알고리즘: Trend Following & Swing</span>
                <span>•</span>
                <span>최종 주문은 본인의 책임하에 진행</span>
            </div>
        </footer>
    </div>

    <script>
        const DATA = __DATA__;
        const REGIME = __REGIME__;
        let currentMode = 'trend';
        let grid = null;

        function updateSummary() {
            document.getElementById('totalCount').innerText = DATA.length + '개 종목';
            const highScores = DATA.filter(r => r[currentMode] >= 80).length;
            document.getElementById('highScoreCount').innerText = highScores + '건 발생';

            if (REGIME.kospi) {
                const badge = document.getElementById('regimeBadge');
                const isBull = REGIME.ok;
                badge.innerHTML = `
                    <div class="flex items-center gap-4 bg-slate-800/80 border border-slate-700 px-5 py-3 rounded-2xl shadow-lg">
                        <div class="flex flex-col">
                            <span class="text-[10px] text-slate-500 font-extrabold uppercase tracking-tighter">Market Regime</span>
                            <span class="text-sm font-bold ${isBull ? 'text-emerald-400' : 'text-rose-400'}">
                                ${isBull ? '🟢 Strong Bull' : '🔴 Bear Market'}
                            </span>
                        </div>
                        <div class="h-10 w-[1px] bg-slate-700/50"></div>
                        <div class="flex flex-col">
                            <span class="text-[10px] text-slate-500 font-extrabold uppercase tracking-tighter">KOSPI vs 200MA</span>
                            <span class="text-sm font-mono font-bold">${REGIME.diff_pct > 0 ? '+' : ''}${REGIME.diff_pct.toFixed(1)}%</span>
                        </div>
                    </div>
                `;
            }
        }

        function renderTable() {
            const sortedData = [...DATA].sort((a, b) => b[currentMode] - a[currentMode]);
            
            const columns = [
                { name: "종목명", formatter: (cell, row) => gridjs.html(`<div class="flex flex-col"><a href="https://finance.naver.com/item/main.naver?code=${row.cells[1].data}" target="_blank" class="font-bold text-blue-400 hover:underline">${row.cells[2].data}</a><span class="text-[10px] text-slate-500 font-mono mt-0.5">${row.cells[1].data}</span></div>`) },
                { name: "코드", hidden: true },
                { name: "이름", hidden: true },
                { name: "현재가", formatter: (cell) => `₩${Number(cell).toLocaleString()}` },
                { name: "전일비", formatter: (cell) => gridjs.html(`<span class="${cell >= 0 ? 'text-emerald-400' : 'text-rose-400'} font-mono text-xs font-bold">${cell > 0 ? '+' : ''}${cell.toFixed(1)}%</span>`) },
                { name: "강도점수", formatter: (cell) => gridjs.html(`<span class="pill ${cell >= 80 ? 'pill-buy' : cell >= 60 ? 'pill-neutral' : 'pill-sell'}">${cell}</span>`) },
                { name: "신호 분석 사유", width: '320px', formatter: (cell) => gridjs.html(`<span class="text-[12px] text-slate-400 leading-relaxed">${cell}</span>`) },
                { name: "손절가 가이드", formatter: (cell) => gridjs.html(`<span class="text-rose-400/90 font-mono text-sm font-medium">₩${Number(cell).toLocaleString()}</span>`) },
                { name: "목표가 가이드", formatter: (cell) => gridjs.html(`<span class="text-emerald-400/90 font-mono text-sm font-medium">₩${Number(cell).toLocaleString()}</span>`) }
            ];

            const tableData = sortedData.map(r => [
                '', r.code, r.name, r.close, r.chg1d, r[currentMode],
                currentMode === 'trend' ? r.trend_why : r.swing_why,
                r.stop, r.target
            ]);

            if (grid) grid.destroy();
            
            grid = new gridjs.Grid({
                columns: columns,
                data: tableData,
                search: true,
                pagination: { limit: 15 },
                sort: true,
                resizable: true,
                language: {
                    'search': { 'placeholder': '종목명 또는 코드 검색...' },
                    'pagination': { 'previous': '이전', 'next': '다음', 'showing': '검색 결과', 'results': () => '개' }
                }
            }).render(document.getElementById("tableWrapper"));
        }

        function changeMode(mode) {
            currentMode = mode;
            document.querySelectorAll('.tab-btn').forEach(btn => {
                btn.classList.remove('bg-blue-600', 'text-white', 'shadow-lg', 'shadow-blue-900/40');
                btn.classList.add('text-slate-400', 'hover:text-white', 'hover:bg-slate-700/50');
            });
            const activeBtn = document.getElementById('btn-' + mode);
            activeBtn.classList.add('bg-blue-600', 'text-white', 'shadow-lg', 'shadow-blue-900/40');
            activeBtn.classList.remove('text-slate-400', 'hover:text-white', 'hover:bg-slate-700/50');
            
            updateSummary();
            renderTable();
        }

        updateSummary();
        renderTable();
    </script>
</body>
</html>
"""
