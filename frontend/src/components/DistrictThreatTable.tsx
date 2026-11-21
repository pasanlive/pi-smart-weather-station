import { useMemo, useState, useId } from "react";
import { ChevronUp, Search } from "lucide-react";
import type { DistrictThreat, ThreatLevel } from "../types/nowcast";

interface DistrictThreatTableProps {
  districts: DistrictThreat[];
  selectedDistrictId: string | null;
  onSelectDistrict: (districtId: string) => void;
}

type SortCol = "name" | "threat_level" | "eta_min" | "peak_coverage_pct" | "wind_gusts_kmh";
type SortDir = "ascending" | "descending";

const THREAT_WEIGHT: Record<ThreatLevel, number> = {
  SEVERE: 4,
  HIGH: 3,
  MODERATE: 2,
  LOW: 1,
};

const THREAT_BADGE: Record<ThreatLevel, string> = {
  LOW: "border-emerald-500/35 bg-emerald-500/15 text-emerald-300",
  MODERATE: "border-yellow-500/45 bg-yellow-500/15 text-yellow-200",
  HIGH: "border-orange-500/50 bg-orange-500/20 text-orange-200",
  SEVERE: "border-rose-500/60 bg-rose-500/25 text-rose-200",
};

const collator = new Intl.Collator(undefined, { numeric: true, sensitivity: "base" });

function formatEta(etaMin: number | null): string {
  if (etaMin === 0) return "NOW";
  if (etaMin !== null) return `+${etaMin}m`;
  return "—";
}

export function DistrictThreatTable({
  districts,
  selectedDistrictId,
  onSelectDistrict,
}: DistrictThreatTableProps) {
  const captionId = useId();
  const [query, setQuery] = useState("");
  const [sortCol, setSortCol] = useState<SortCol>("threat_level");
  const [sortDir, setSortDir] = useState<SortDir>("descending");
  const [announcement, setAnnouncement] = useState("");

  const handleSort = (col: SortCol, label: string) => {
    const nextDir: SortDir =
      sortCol === col ? (sortDir === "ascending" ? "descending" : "ascending") : "ascending";
    setSortCol(col);
    setSortDir(nextDir);
    setAnnouncement(`Sorted districts by ${label}, ${nextDir}, ${filteredSorted.length} rows`);
  };

  const filteredSorted = useMemo(() => {
    const q = query.trim().toLowerCase();
    const list = q ? districts.filter((d) => d.name.toLowerCase().includes(q)) : [...districts];
    const factor = sortDir === "ascending" ? 1 : -1;

    list.sort((a, b) => {
      switch (sortCol) {
        case "name":
          return factor * collator.compare(a.name, b.name);
        case "threat_level": {
          const diff = THREAT_WEIGHT[a.threat_level] - THREAT_WEIGHT[b.threat_level];
          if (diff !== 0) return factor * diff;
          const ea = a.eta_min ?? 999;
          const eb = b.eta_min ?? 999;
          if (ea !== eb) return ea - eb;
          return b.peak_coverage_pct - a.peak_coverage_pct;
        }
        case "eta_min": {
          const ea = a.eta_min ?? 999;
          const eb = b.eta_min ?? 999;
          return factor * (ea - eb);
        }
        case "peak_coverage_pct":
          return factor * (a.peak_coverage_pct - b.peak_coverage_pct);
        case "wind_gusts_kmh":
          return factor * (a.wind_gusts_kmh - b.wind_gusts_kmh);
      }
    });
    return list;
  }, [districts, query, sortCol, sortDir]);

  const ariaSort = (col: SortCol): SortDir | undefined => (sortCol === col ? sortDir : undefined);

  return (
    <section className="flex min-h-0 flex-1 flex-col rounded-lg border border-slate-800 bg-slate-900/75 p-3">
      <div className="mb-2 flex items-center justify-between gap-2">
        <h2 id={captionId} className="text-xs font-semibold uppercase tracking-wider text-slate-200">
          District Threat Matrix ({districts.length})
        </h2>
        <div className="relative w-36">
          <Search className="pointer-events-none absolute left-2 top-1/2 h-3 w-3 -translate-y-1/2 text-slate-500" />
          <input
            type="search"
            value={query}
            onChange={(e) => setQuery(e.target.value)}
            placeholder="Filter district..."
            aria-label="Filter districts"
            className="w-full rounded border border-slate-700 bg-slate-950 py-1 pl-6 pr-2 text-[11px] text-slate-100 placeholder:text-slate-500 focus:border-cyan-400 focus:outline-none"
          />
        </div>
      </div>

      {/* Single visually hidden polite status element per sortable-table guidance */}
      <div role="status" className="sr-status">
        {announcement}
      </div>

      <div
        role="region"
        aria-labelledby={captionId}
        tabIndex={0}
        className="min-h-0 flex-1 overflow-auto rounded border border-slate-800/80 bg-slate-950/60"
      >
        <table className="sortable-table w-full min-w-[360px] border-collapse text-left text-[11px]">
          <caption className="sr-status">Sri Lanka 25-District Weather Threat Table</caption>
          <thead className="sticky top-0 z-10 border-b border-slate-800 bg-slate-900 text-[10px] font-semibold uppercase tracking-wider text-slate-400">
            <tr>
              <th scope="col" aria-sort={ariaSort("name")}>
                <button type="button" onClick={() => handleSort("name", "District")}>
                  <span>District</span>
                  <ChevronUp className="sort-indicator h-3 w-3 shrink-0" aria-hidden="true" />
                </button>
              </th>
              <th scope="col" aria-sort={ariaSort("threat_level")}>
                <button type="button" onClick={() => handleSort("threat_level", "Threat")}>
                  <span>Threat</span>
                  <ChevronUp className="sort-indicator h-3 w-3 shrink-0" aria-hidden="true" />
                </button>
              </th>
              <th scope="col" aria-sort={ariaSort("eta_min")}>
                <button type="button" onClick={() => handleSort("eta_min", "ETA")}>
                  <span>ETA</span>
                  <ChevronUp className="sort-indicator h-3 w-3 shrink-0" aria-hidden="true" />
                </button>
              </th>
              <th scope="col" aria-sort={ariaSort("peak_coverage_pct")}>
                <button type="button" onClick={() => handleSort("peak_coverage_pct", "Peak Cover")}>
                  <span>Peak</span>
                  <ChevronUp className="sort-indicator h-3 w-3 shrink-0" aria-hidden="true" />
                </button>
              </th>
              <th scope="col" aria-sort={ariaSort("wind_gusts_kmh")}>
                <button type="button" onClick={() => handleSort("wind_gusts_kmh", "Rain and Wind")}>
                  <span>Rain / Wind</span>
                  <ChevronUp className="sort-indicator h-3 w-3 shrink-0" aria-hidden="true" />
                </button>
              </th>
            </tr>
          </thead>
          <tbody className="divide-y divide-slate-800/60">
            {filteredSorted.map((d) => {
              const isSelected = selectedDistrictId === d.district_id;
              return (
                <tr
                  key={d.district_id}
                  onClick={() => onSelectDistrict(d.district_id)}
                  className={`cursor-pointer transition ${
                    isSelected ? "bg-cyan-950/45" : "hover:bg-slate-900/75"
                  }`}
                >
                  <td className="px-2 py-1.5 font-medium text-slate-100">{d.name}</td>
                  <td className="px-2 py-1.5">
                    <span
                      className={`inline-block rounded border px-1.5 py-0.5 font-mono text-[9px] font-bold ${THREAT_BADGE[d.threat_level]}`}
                    >
                      {d.threat_level}
                    </span>
                  </td>
                  <td className="px-2 py-1.5 font-mono text-[10px]">
                    <span
                      className={`rounded px-1.5 py-0.5 ${
                        d.eta_min === 0
                          ? "bg-rose-500/20 font-semibold text-rose-200"
                          : d.eta_min !== null
                          ? "bg-cyan-500/15 font-semibold text-cyan-200"
                          : "text-slate-500"
                      }`}
                    >
                      {formatEta(d.eta_min)}
                    </span>
                  </td>
                  <td className="px-2 py-1.5 font-mono text-[10px] text-slate-300">
                    {d.peak_coverage_pct.toFixed(0)}%
                  </td>
                  <td className="px-2 py-1.5 font-mono text-[10px] text-slate-400">
                    {d.precipitation_mm.toFixed(1)}mm • {d.wind_gusts_kmh.toFixed(0)}k
                  </td>
                </tr>
              );
            })}
          </tbody>
        </table>
      </div>
    </section>
  );
}
