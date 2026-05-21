import { useState, useRef, useEffect, useCallback } from "react";

// ─── Backend API ───
// Le solveur CP-SAT et le LLM tournent côté Python (voir api_server.py).
// Ce frontend ne fait QUE de l'affichage et relaie les messages.

const API_BASE = import.meta.env?.VITE_API_BASE || "http://127.0.0.1:8000";

async function callChat(sessionId, message) {
  const body = { session_id: sessionId, message };

  const resp = await fetch(`${API_BASE}/chat`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(body),
  });
  if (!resp.ok) throw new Error(`Backend error ${resp.status}`);
  return resp.json();
}

async function callReset(sessionId) {
  await fetch(`${API_BASE}/reset`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ session_id: sessionId }),
  });
}

async function callState(sessionId) {
  const resp = await fetch(`${API_BASE}/state?session_id=${encodeURIComponent(sessionId)}`);
  if (!resp.ok) return null;
  return resp.json();
}

// ─── Theming ───

const CATEGORY_COLORS = {
  culture: { bg: "#E8D5B7", text: "#5C4033", accent: "#A0785A" },
  gastro: { bg: "#F2D4D4", text: "#8B2500", accent: "#CD5C5C" },
  nature: { bg: "#D4E8D4", text: "#2E5C2E", accent: "#6B8E6B" },
  shopping: { bg: "#D4D4E8", text: "#2E2E5C", accent: "#6B6B8E" },
  nightlife: { bg: "#E8D4E8", text: "#5C2E5C", accent: "#8E6B8E" },
};

const CATEGORY_ICONS = {
  culture: "🏛️",
  gastro: "🍷",
  nature: "🌿",
  shopping: "🛍️",
  nightlife: "🌙",
};

// ─── Components ───

function ConstraintBadge({ label, value }) {
  return (
    <span style={{
      display: "inline-flex", alignItems: "center", gap: 4,
      padding: "3px 10px", borderRadius: 20,
      background: "rgba(160,120,90,0.12)", color: "#5C4033",
      fontSize: 12, fontFamily: "'Instrument Serif', Georgia, serif",
      border: "1px solid rgba(160,120,90,0.2)",
    }}>
      {label}: <strong>{Array.isArray(value) ? value.join(", ") : String(value)}</strong>
    </span>
  );
}

const TRANSPORT_LABELS = {
  foot:        { icon: "🚶", label: "À pied",     color: "#8B6F4E" },
  metro:       { icon: "🚇", label: "Métro",      color: "#003189" },
  bus:         { icon: "🚌", label: "Bus",         color: "#1A7A1A" },
  rer:         { icon: "🚆", label: "RER",         color: "#6B3FA0" },
  tram:        { icon: "🚊", label: "Tramway",     color: "#9B1C1C" },
  train:       { icon: "🚂", label: "Train",       color: "#1A5276" },
  funiculaire: { icon: "🚡", label: "Funiculaire", color: "#7D3C00" },
  ferry:       { icon: "⛴",  label: "Ferry",       color: "#154360" },
  navette:     { icon: "🚐", label: "Navette",     color: "#2E4057" },
  transit:     { icon: "🚇", label: "Transports",  color: "#555" },
  bike:        { icon: "🚲", label: "Vélo",        color: "#6B8E23" },
  car:         { icon: "🚗", label: "Voiture",     color: "#8B0000" },
};

function formatDistance(meters) {
  if (meters == null) return null;
  if (meters < 1000) return `${meters} m`;
  return `${(meters / 1000).toFixed(1)} km`;
}



function TransitionRow({ transition }) {
  if (!transition) return null;
  const t = TRANSPORT_LABELS[transition.mode] || TRANSPORT_LABELS.foot;
  const dist = formatDistance(transition.distance_m);
  const color = t.color || "#8B6F4E";

  return (
    <div style={{
      padding: "4px 0 4px 60px", marginBottom: 2,
      fontSize: 11, color: "#8B6F4E",
      fontFamily: "'DM Mono', monospace",
    }}>
      <div style={{ display: "flex", alignItems: "center", gap: 8 }}>
        {/* Icône colorée selon le mode */}
        <span style={{
          fontSize: 13,
          background: `${color}18`,
          border: `1px solid ${color}44`,
          borderRadius: 6,
          padding: "1px 5px",
          color,
        }}>
          {t.icon}
        </span>

        <span style={{
          flex: 1, height: 1,
          background: "repeating-linear-gradient(to right, #C8B89C 0 4px, transparent 4px 8px)",
        }} />

        <span style={{ fontWeight: 600 }}>{transition.minutes} min</span>
        {dist && <span style={{ opacity: 0.6 }}>· {dist}</span>}

        {/* Badge mode */}
        <span style={{
          background: `${color}18`, color,
          border: `1px solid ${color}55`,
          borderRadius: 8, padding: "1px 7px",
          fontSize: 10, fontWeight: 600,
        }}>
          {t.label}
        </span>

        {/* Bouton Google Maps pour tous les modes */}
        {transition.maps_url && (
          <a
            href={transition.maps_url}
            target="_blank"
            rel="noopener noreferrer"
            style={{
              display: "inline-flex", alignItems: "center", gap: 3,
              padding: "2px 8px", borderRadius: 10,
              background: "#1a73e8", color: "#fff",
              fontSize: 10, fontWeight: 600, textDecoration: "none",
              whiteSpace: "nowrap",
            }}
          >
            🗺 Itinéraire
          </a>
        )}
      </div>
    </div>
  );
}

function HotelCard({ hotel }) {
  if (!hotel) return null;
  return (
    <div style={{
      margin: "10px 20px", padding: "12px 16px", borderRadius: 12,
      background: "linear-gradient(135deg, #FFFAF2, #F5E9D7)",
      border: "1px solid #D8C5A8",
      display: "flex", gap: 14, alignItems: "flex-start",
    }}>
      <div style={{
        fontSize: 30, lineHeight: 1, marginTop: 2,
      }}>🏨</div>
      <div style={{ flex: 1, minWidth: 0 }}>
        <div style={{
          display: "flex", justifyContent: "space-between", alignItems: "baseline",
          gap: 10, marginBottom: 2,
        }}>
          <div style={{
            fontSize: 15, fontWeight: 600, color: "#3C2415",
            fontFamily: "'Instrument Serif', Georgia, serif",
          }}>
            {hotel.name}
            {hotel.stars > 0 && (
              <span style={{ marginLeft: 6, color: "#C19A4D", fontSize: 12 }}>
                {"★".repeat(hotel.stars)}
              </span>
            )}
          </div>
          <span style={{
            fontSize: 13, color: "#5C4033", fontWeight: 600,
            fontFamily: "'DM Mono', monospace", whiteSpace: "nowrap",
          }}>{hotel.price_per_night}€<span style={{ opacity: 0.6, fontWeight: 400 }}> /nuit</span></span>
        </div>
        {hotel.address && (
          <div style={{
            fontSize: 11, color: "#8B6F4E", marginBottom: 4,
            fontFamily: "'DM Mono', monospace",
          }}>📍 {hotel.address}</div>
        )}
        {hotel.description && (
          <div style={{
            fontSize: 12, color: "#5C4033", opacity: 0.85,
            fontFamily: "'Instrument Serif', Georgia, serif",
            fontStyle: "italic",
          }}>{hotel.description}</div>
        )}
      </div>
    </div>
  );
}

function TimelineActivity({ activity, travelers }) {
  const cat = CATEGORY_COLORS[activity.category] || CATEGORY_COLORS.culture;
  const icon = CATEGORY_ICONS[activity.category] || "📍";
  const cost = activity.cost ?? activity.cost_euros ?? 0;

  return (
    <div style={{ display: "flex", gap: 12, alignItems: "stretch", marginBottom: 2 }}>
      <div style={{
        display: "flex", flexDirection: "column", alignItems: "center",
        width: 56, flexShrink: 0,
      }}>
        <span style={{
          fontSize: 13, fontWeight: 600, color: "#5C4033",
          fontFamily: "'DM Mono', monospace",
        }}>{activity.start_time}</span>
        <div style={{
          width: 2, flex: 1, background: `linear-gradient(to bottom, ${cat.accent}, transparent)`,
          marginTop: 4,
        }} />
        <span style={{
          fontSize: 11, color: "#999",
          fontFamily: "'DM Mono', monospace",
        }}>{activity.end_time}</span>
      </div>

      <div style={{
        flex: 1, padding: "10px 14px", borderRadius: 10,
        background: cat.bg, border: `1px solid ${cat.accent}33`,
        position: "relative", overflow: "hidden",
      }}>
        <div style={{
          position: "absolute", top: 8, right: 10,
          fontSize: 20, opacity: 0.3,
        }}>{icon}</div>
        <div style={{
          fontSize: 14, fontWeight: 600, color: cat.text,
          fontFamily: "'Instrument Serif', Georgia, serif",
          marginBottom: 2,
        }}>{activity.name}</div>
        {activity.address && (
          <div style={{
            fontSize: 10, color: cat.text, opacity: 0.65,
            fontFamily: "'DM Mono', monospace",
            marginBottom: 3,
          }}>
            📍 {activity.address}{activity.zone ? ` (${activity.zone})` : ""}
          </div>
        )}
        <div style={{
          display: "flex", gap: 10, fontSize: 11, color: cat.text, opacity: 0.7,
          fontFamily: "'DM Mono', monospace",
        }}>
          <span>{activity.duration_hours ?? activity.duration}h</span>
          {!activity.address && activity.zone && <span>📍 {activity.zone}</span>}
          {cost > 0 && <span>{cost}€</span>}
          {cost === 0 && <span style={{ color: "#2E5C2E" }}>Gratuit</span>}
        </div>
      </div>
    </div>
  );
}

const WEEKDAY_FR = ["Lundi", "Mardi", "Mercredi", "Jeudi", "Vendredi", "Samedi", "Dimanche"];

function formatDayHeader(dayIndex1, startDate, weekdays) {
  if (!startDate) return `Jour ${dayIndex1}`;
  // dayIndex1 est 1-indexé, weekdays est 0-indexé
  const idx = dayIndex1 - 1;
  const wd = (weekdays && weekdays[idx] != null) ? WEEKDAY_FR[weekdays[idx]] : "";
  // Calcul de la date réelle
  try {
    const [y, m, d] = startDate.split("-").map(Number);
    const dt = new Date(Date.UTC(y, m - 1, d + idx));
    const dd = String(dt.getUTCDate()).padStart(2, "0");
    const mm = String(dt.getUTCMonth() + 1).padStart(2, "0");
    return wd ? `${wd} ${dd}/${mm}` : `${dd}/${mm}`;
  } catch {
    return wd || `Jour ${dayIndex1}`;
  }
}

function DayCard({ day, travelers, startDate, weekdays }) {
  if (!day) return null;
  const transitions = day.transitions || [];
  const heading = formatDayHeader(day.day, startDate, weekdays);
  return (
    <div style={{ padding: 16 }}>
      <div style={{
        display: "flex", justifyContent: "space-between", alignItems: "center",
        marginBottom: 12,
      }}>
        <h3 style={{
          margin: 0, fontSize: 18,
          fontFamily: "'Instrument Serif', Georgia, serif",
          color: "#3C2415",
        }}>{heading}</h3>
        <div style={{
          display: "flex", gap: 10, alignItems: "center",
          fontSize: 12, color: "#A0785A",
          fontFamily: "'DM Mono', monospace",
        }}>
          <span>{day.activities.length} activités</span>
          {day.total_travel_minutes != null && day.total_travel_minutes > 0 && (
            <span title="Temps de trajet total entre activités">
              · 🚶 {day.total_travel_minutes} min trajet
            </span>
          )}
        </div>
      </div>
      <div style={{ display: "flex", flexDirection: "column", gap: 2 }}>
        {day.activities.length === 0 && (
          <div style={{ padding: 16, textAlign: "center", color: "#999" }}>
            Rien de prévu ce jour (le solveur a jugé qu'un jour de repos respectait mieux tes contraintes).
          </div>
        )}
        {day.activities.map((act, i) => (
          <div key={`${act.id}-${i}`}>
            <TimelineActivity activity={act} travelers={travelers} />
            {i < transitions.length && <TransitionRow transition={transitions[i]} />}
          </div>
        ))}
      </div>
    </div>
  );
}

function BudgetBar({ summary }) {
  if (!summary || !summary.budget) return null;
  const segments = [
    { label: "Hôtel", value: summary.hotel_cost || 0, color: "#A0785A" },
    { label: "Repas", value: summary.food_cost || 0, color: "#CD5C5C" },
    { label: "Activités", value: summary.activity_cost || 0, color: "#6B8E6B" },
  ];

  return (
    <div style={{ padding: "14px 0" }}>
      <div style={{
        display: "flex", justifyContent: "space-between", marginBottom: 8,
        fontFamily: "'DM Mono', monospace", fontSize: 13,
      }}>
        <span style={{ color: "#5C4033" }}>{summary.total_cost}€ / {summary.budget}€</span>
        <span style={{ color: summary.remaining_budget > 0 ? "#2E5C2E" : "#8B2500" }}>
          {summary.remaining_budget > 0 ? `${summary.remaining_budget}€ restants` : "Budget dépassé !"}
        </span>
      </div>
      <div style={{
        height: 10, borderRadius: 10, background: "#F5F0EB",
        overflow: "hidden", display: "flex",
      }}>
        {segments.map((seg, i) => (
          <div key={i} style={{
            width: `${(seg.value / summary.budget) * 100}%`,
            background: seg.color, transition: "width 0.5s ease",
          }} />
        ))}
      </div>
      <div style={{
        display: "flex", gap: 14, marginTop: 6,
        fontSize: 11, fontFamily: "'DM Mono', monospace",
      }}>
        {segments.map((seg, i) => (
          <span key={i} style={{ display: "flex", alignItems: "center", gap: 4 }}>
            <span style={{
              width: 8, height: 8, borderRadius: "50%",
              background: seg.color, display: "inline-block",
            }} />
            {seg.label}: {seg.value}€
          </span>
        ))}
      </div>
    </div>
  );
}

function StatsPanel({ stats, city, dataSource }) {
  if (!stats && !city) return null;
  return (
    <div style={{
      display: "flex", gap: 16, padding: "10px 0", flexWrap: "wrap",
      fontSize: 11, fontFamily: "'DM Mono', monospace", color: "#999",
      borderTop: "1px solid #eee",
    }}>
      {city?.name && <span>🏙️ {city.name}{city.country ? ` (${city.country})` : ""}</span>}
      {stats?.solve_time_ms != null && <span>⚡ CP-SAT {stats.solve_time_ms}ms</span>}
      {stats?.branches != null && <span>🌳 {stats.branches} branches</span>}
      {stats?.conflicts != null && <span>💥 {stats.conflicts} conflits</span>}
      {stats?.status_name && <span>📊 {stats.status_name}</span>}
      {dataSource && <span>📡 {dataSource}</span>}
    </div>
  );
}

function ChatMessage({ message }) {
  const isUser = message.role === "user";
  return (
    <div style={{
      display: "flex", justifyContent: isUser ? "flex-end" : "flex-start",
      marginBottom: 10,
    }}>
      <div style={{
        maxWidth: "85%", padding: "10px 14px", borderRadius: 16,
        background: isUser ? "#3C2415" : "#F5F0EB",
        color: isUser ? "#F5F0EB" : "#3C2415",
        fontSize: 14, lineHeight: 1.5,
        fontFamily: "'Instrument Serif', Georgia, serif",
        borderBottomRightRadius: isUser ? 4 : 16,
        borderBottomLeftRadius: isUser ? 16 : 4,
      }}>
        {message.content}
        {message.extracted && Object.keys(message.extracted).length > 0 && (
          <div style={{
            marginTop: 8, paddingTop: 8,
            borderTop: `1px solid ${isUser ? "rgba(255,255,255,0.15)" : "rgba(0,0,0,0.08)"}`,
            display: "flex", flexWrap: "wrap", gap: 4,
          }}>
            {Object.entries(message.extracted).map(([k, v]) => (
              <span key={k} style={{
                fontSize: 11, padding: "2px 8px", borderRadius: 10,
                background: isUser ? "rgba(255,255,255,0.15)" : "rgba(160,120,90,0.12)",
                fontFamily: "'DM Mono', monospace",
              }}>
                {k}: {Array.isArray(v) ? v.join(",") : String(v)}
              </span>
            ))}
          </div>
        )}
        {message.errors && message.errors.length > 0 && (
          <div style={{
            marginTop: 6, fontSize: 11, color: "#8B2500",
            fontFamily: "'DM Mono', monospace",
          }}>
            ⚠ {message.errors.join(" · ")}
          </div>
        )}
      </div>
    </div>
  );
}

// ─── Main App ───

export default function TravelPlannerApp() {
  const [sessionId] = useState(() => `s-${Date.now()}-${Math.random().toString(36).slice(2, 8)}`);
  const [messages, setMessages] = useState([{
    role: "assistant",
    content: "Bonjour, je suis votre assistant de planification, comment puis-je vous aider à planifier votre séjour ?",
  }]);
  const [input, setInput] = useState("");
  const [constraints, setConstraints] = useState(null);
  const [plan, setPlan] = useState(null);
  const [city, setCity] = useState(null);
  const [hotel, setHotel] = useState(null);
  const [dataSource, setDataSource] = useState("");
  const [activeDay, setActiveDay] = useState(0);
  const [loading, setLoading] = useState(false);
  const [showConstraints, setShowConstraints] = useState(false);
  const chatEndRef = useRef(null);

  useEffect(() => {
    chatEndRef.current?.scrollIntoView({ behavior: "smooth" });
  }, [messages]);

  // Charge l'état initial du serveur (contraintes par défaut)
  useEffect(() => {
    callState(sessionId).then(s => {
      if (s?.constraints) setConstraints(s.constraints);
    }).catch(() => { });
  }, [sessionId]);

  const handleSend = useCallback(async () => {
    if (!input.trim() || loading) return;
    const userMsg = input.trim();
    setInput("");
    setMessages(prev => [...prev, { role: "user", content: userMsg }]);
    setLoading(true);

    try {
      const result = await callChat(sessionId, userMsg);

      setConstraints(result.constraints);
      if (result.plan) setPlan(result.plan);
      if (result.city) setCity(result.city);
      if (result.plan?.data_source) setDataSource(result.plan.data_source);
      if (result.plan?.hotel) setHotel(result.plan.hotel);
      setActiveDay(0);

      setMessages(prev => [
        ...prev.slice(0, -1),
        { role: "user", content: userMsg, extracted: result.extracted },
        { role: "assistant", content: result.reply, errors: result.errors },
      ]);
    } catch (e) {
      setMessages(prev => [...prev, {
        role: "assistant",
        content: `Erreur côté backend : ${e.message}. Le serveur FastAPI tourne-t-il sur ${API_BASE} ?`,
      }]);
    }
    setLoading(false);
  }, [input, sessionId, loading]);

  const handleReset = useCallback(async () => {
    await callReset(sessionId);
    const s = await callState(sessionId);
    setConstraints(s?.constraints || null);
    setPlan(null);
    setCity(null);
    setHotel(null);
    setMessages([{
      role: "assistant",
      content: "Nouvelle session. Dis-moi ce que tu veux planifier !",
    }]);
  }, [sessionId]);

  const suggestions = [
    "5 jours à Rome, budget 1500€, on adore la culture",
    "Je déteste le shopping, plus de gastronomie !",
    "On est 2, rythme tranquille svp",
    "Je veux absolument voir le Colisée et le Vatican",
    "Jour 3 plus relax, max 2 activités",
    "Budget serré : 1000€ pour 3 jours",
  ];

  const activeDayData = plan?.days?.[activeDay];

  return (
    <div style={{
      display: "flex", height: "100vh", width: "100%",
      fontFamily: "'Instrument Serif', Georgia, serif",
      background: "#FDFBF7",
      overflow: "hidden",
    }}>
      {/* ─── Gauche : Chat ─── */}
      <div style={{
        width: "40%", minWidth: 340,
        display: "flex", flexDirection: "column",
        borderRight: "1px solid #E8E0D8",
        background: "#FFFFFF",
      }}>
        <div style={{
          padding: "16px 20px",
          borderBottom: "1px solid #E8E0D8",
          background: "#3C2415",
          color: "#F5F0EB",
          display: "flex", justifyContent: "space-between", alignItems: "center",
        }}>
          <div>
            <h1 style={{
              margin: 0, fontSize: 20,
              fontFamily: "'Instrument Serif', Georgia, serif",
              fontWeight: 400,
            }}>
              🏛️ Planificateur CP-SAT
            </h1>
            <p style={{
              margin: "4px 0 0", fontSize: 12, opacity: 0.7,
              fontFamily: "'DM Mono', monospace",
            }}>LLM (qwen3) + CP-SAT + OpenTripMap/OSRM</p>
          </div>
          <button
            onClick={handleReset}
            style={{
              background: "transparent", border: "1px solid rgba(245,240,235,0.3)",
              color: "#F5F0EB", padding: "4px 10px", borderRadius: 8,
              fontSize: 11, cursor: "pointer",
              fontFamily: "'DM Mono', monospace",
            }}
          >reset</button>
        </div>

        <div style={{
          flex: 1, overflowY: "auto", padding: "12px 16px 8px",
        }}>
          {messages.map((msg, i) => <ChatMessage key={i} message={msg} />)}
          {loading && (
            <div style={{ display: "flex", justifyContent: "flex-start", marginBottom: 10 }}>
              <div style={{
                padding: "10px 18px", borderRadius: 16, borderBottomLeftRadius: 4,
                background: "#F5F0EB", fontSize: 14,
              }}>
                <span style={{ animation: "pulse 1.5s infinite" }}>
                  Extraction LLM → fetch POI → résolution CP-SAT…
                </span>
              </div>
            </div>
          )}
          <div ref={chatEndRef} />
        </div>

        {messages.length < 3 && (
          <div style={{
            padding: "0 16px 8px",
            display: "flex", flexWrap: "wrap", gap: 6,
          }}>
            {suggestions.slice(0, 4).map((s, i) => (
              <button key={i} onClick={() => setInput(s)}
                style={{
                  padding: "6px 12px", borderRadius: 20,
                  background: "#F5F0EB", border: "1px solid #E8E0D8",
                  fontSize: 12, color: "#5C4033", cursor: "pointer",
                  fontFamily: "'Instrument Serif', Georgia, serif",
                }}
              >{s}</button>
            ))}
          </div>
        )}


        <div style={{
          padding: "8px 16px 12px",
          display: "flex",
          gap: 8,
        }}>
          <input
            value={input}
            onChange={e => setInput(e.target.value)}
            onKeyDown={e => e.key === "Enter" && handleSend()}
            placeholder="Décris ton voyage idéal…"
            disabled={loading}
            style={{
              flex: 1, padding: "10px 14px", borderRadius: 12,
              border: "1px solid #E8E0D8", background: "#FDFBF7",
              fontSize: 14, fontFamily: "'Instrument Serif', Georgia, serif",
              outline: "none", color: "#3C2415",
            }}
          />
          <button
            onClick={handleSend}
            disabled={loading || !input.trim()}
            style={{
              padding: "10px 18px", borderRadius: 12,
              background: loading ? "#ccc" : "#3C2415",
              color: "#F5F0EB", border: "none",
              fontSize: 14, cursor: loading ? "default" : "pointer",
              fontFamily: "'DM Mono', monospace",
            }}
          >→</button>
        </div>
      </div>

      {/* ─── Droite : Plan ─── */}
      <div style={{
        flex: 1, display: "flex", flexDirection: "column",
        overflowY: "auto", background: "#FDFBF7",
      }}>
        <div style={{
          padding: "12px 20px",
          borderBottom: "1px solid #E8E0D8",
          background: "#FFFFFF",
        }}>
          <div style={{
            display: "flex", justifyContent: "space-between", alignItems: "center",
            marginBottom: showConstraints ? 10 : 0,
          }}>
            <span style={{
              fontSize: 13, color: "#A0785A",
              fontFamily: "'DM Mono', monospace",
            }}>
              Contraintes actives ({constraints ? Object.entries(constraints).filter(([, v]) =>
                v !== undefined && v !== null && (Array.isArray(v) ? v.length > 0 : true)
              ).length : 0})
            </span>
            <button
              onClick={() => setShowConstraints(!showConstraints)}
              style={{
                background: "none", border: "none", cursor: "pointer",
                fontSize: 12, color: "#A0785A",
                fontFamily: "'DM Mono', monospace",
              }}
            >
              {showConstraints ? "▲ Masquer" : "▼ Voir"}
            </button>
          </div>
          {showConstraints && constraints && (
            <div style={{ display: "flex", flexWrap: "wrap", gap: 6 }}>
              {Object.entries(constraints).map(([k, v]) => {
                if (v === null || v === undefined) return null;
                if (Array.isArray(v) && v.length === 0) return null;
                return <ConstraintBadge key={k} label={k} value={v} />;
              })}
            </div>
          )}
        </div>

        {plan?.status === "INFEASIBLE" && (
          <div style={{
            margin: "16px 20px", padding: 16, borderRadius: 10,
            background: "#FFF2E0", color: "#8B2500",
            fontFamily: "'Instrument Serif', Georgia, serif",
          }}>
            <strong>Plan infaisable.</strong> {plan.message}
          </div>
        )}

        {plan?.summary && (
          <div style={{ padding: "0 20px" }}>
            <BudgetBar summary={plan.summary} />
          </div>
        )}

        {hotel && (
          <HotelCard hotel={hotel} />
        )}

        {plan?.days && plan.days.length > 0 && (() => {
          // Compter les modes effectifs utilisés dans le plan
          const modeCounts = {};
          for (const d of plan.days) {
            for (const t of (d.transitions || [])) {
              modeCounts[t.mode] = (modeCounts[t.mode] || 0) + 1;
            }
          }
          const used = Object.entries(modeCounts);
          return (
            <div style={{
              padding: "0 20px 4px",
              display: "flex", gap: 8, alignItems: "center", flexWrap: "wrap",
              fontSize: 11, color: "#8B6F4E",
              fontFamily: "'DM Mono', monospace",
            }}>
              <span>Transports utilisés :</span>
              {used.length === 0 ? (
                <span style={{ opacity: 0.6 }}>aucun trajet</span>
              ) : (
                used.map(([mode, n]) => {
                  const lbl = TRANSPORT_LABELS[mode] || { icon: "❓", label: mode };
                  return (
                    <span key={mode} style={{
                      padding: "2px 10px", borderRadius: 12,
                      background: "rgba(160,120,90,0.12)",
                      border: "1px solid rgba(160,120,90,0.25)",
                    }}>
                      {lbl.icon} {lbl.label} <span style={{ opacity: 0.6 }}>×{n}</span>
                    </span>
                  );
                })
              )}
            </div>
          );
        })()}

        {plan?.days && plan.days.length > 0 && (
          <div style={{
            display: "flex", gap: 4, padding: "0 20px 8px",
            borderBottom: "1px solid #E8E0D8", flexWrap: "wrap",
          }}>
            {plan.days.map((d, i) => {
              const dayLabel = plan?.start_date
                ? formatDayHeader(d.day, plan.start_date, plan.trip_weekdays).split(" ")[0].slice(0, 3)
                : `J${d.day}`;
              const dateSuffix = plan?.start_date
                ? formatDayHeader(d.day, plan.start_date, plan.trip_weekdays).split(" ")[1] || ""
                : "";
              return (
                <button key={i}
                  onClick={() => setActiveDay(i)}
                  style={{
                    padding: "6px 14px", borderRadius: 8,
                    background: activeDay === i ? "#3C2415" : "transparent",
                    color: activeDay === i ? "#F5F0EB" : "#5C4033",
                    border: activeDay === i ? "none" : "1px solid #E8E0D8",
                    fontSize: 12, cursor: "pointer",
                    fontFamily: "'DM Mono', monospace",
                    lineHeight: 1.3,
                  }}
                >
                  {dayLabel}{dateSuffix && <> <span style={{ opacity: 0.7 }}>{dateSuffix}</span></>}
                  <span style={{ opacity: 0.5, marginLeft: 4 }}>({d.activities.length})</span>
                </button>
              );
            })}
          </div>
        )}

        {activeDayData && (
          <div style={{ padding: "0 20px", flex: 1 }}>
            <DayCard
              day={activeDayData}
              travelers={constraints?.num_travelers || 1}
              startDate={plan?.start_date}
              weekdays={plan?.trip_weekdays}
            />
          </div>
        )}

        {!plan && (
          <div style={{
            padding: 32, textAlign: "center", color: "#A0785A",
            fontFamily: "'Instrument Serif', Georgia, serif",
          }}>
            Commence par décrire ton voyage dans le chat à gauche.
          </div>
        )}

        <div style={{ padding: "0 20px 12px" }}>
          <StatsPanel stats={plan?.stats} city={city} dataSource={dataSource} />
        </div>
      </div>

      <style>{`
        @import url('https://fonts.googleapis.com/css2?family=Instrument+Serif:ital@0;1&family=DM+Mono:wght@300;400;500&display=swap');
        @keyframes pulse {
          0%, 100% { opacity: 1; }
          50% { opacity: 0.5; }
        }
        * { box-sizing: border-box; }
        ::-webkit-scrollbar { width: 6px; }
        ::-webkit-scrollbar-thumb { background: #D4C8BB; border-radius: 3px; }
      `}</style>
    </div>
  );
}
