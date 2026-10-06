import { useState, useRef, useEffect } from 'react';
import axios from 'axios';
import { MessageCircle, X, Send, Loader2, Sparkles, Cpu, ListChecks } from 'lucide-react';

const API = `${process.env.REACT_APP_BACKEND_URL}/api`;

const SUGERENCIAS_INICIALES = [
  '¿Cuánto he gastado en lácteos este mes?',
  '¿Cuántas veces compro yogures a la semana?',
  '¿Cuál ha sido mi compra más cara del mes?',
  '¿De qué categoría compro más productos?',
  '¿Cuál es el producto que más he comprado?',
  '¿Qué día de la semana gasto más?',
  '¿Cuántas devoluciones he hecho?',
];

// Convierte **negrita** en <strong>
function TextoConNegrita({ texto }) {
  return texto.split(/(\*\*[^*]+\*\*)/g).map((parte, i) =>
    parte.startsWith('**') ? <strong key={i} className="text-white">{parte.slice(2, -2)}</strong> : <span key={i}>{parte}</span>
  );
}

function Mensaje({ m, onSugerencia }) {
  if (m.rol === 'usuario') {
    return (
      <div className="flex justify-end">
        <div className="max-w-[85%] bg-blue-600 text-white rounded-2xl rounded-br-sm px-4 py-2 text-sm">{m.texto}</div>
      </div>
    );
  }
  const i = m.interpretacion;
  return (
    <div className="flex justify-start">
      <div className="max-w-[92%] bg-slate-800 text-slate-200 rounded-2xl rounded-bl-sm px-4 py-3 text-sm space-y-2">
        <p className="leading-relaxed"><TextoConNegrita texto={m.texto} /></p>

        {m.tabla && (
          <div className="overflow-x-auto rounded-lg border border-slate-700">
            <table className="w-full text-xs">
              <thead className="bg-slate-900 text-slate-400">
                <tr>{m.tabla.columnas.map((c) => <th key={c} className="px-2 py-1.5 text-left font-medium">{c}</th>)}</tr>
              </thead>
              <tbody>
                {m.tabla.filas.map((fila, r) => (
                  <tr key={r} className="border-t border-slate-700/60">
                    {fila.map((celda, c) => (
                      <td key={c} className={`px-2 py-1.5 ${c > 0 ? 'font-mono whitespace-nowrap' : ''}`}>{celda}</td>
                    ))}
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
        )}

        {m.nota && <p className="text-xs text-slate-400">{m.nota}</p>}

        {m.sugerencias && (
          <div className="flex flex-wrap gap-1.5 pt-1">
            {m.sugerencias.map((s) => (
              <button key={s} onClick={() => onSugerencia(s)}
                className="text-xs bg-slate-900 border border-slate-700 hover:border-blue-500 text-slate-300 rounded-full px-3 py-1 text-left">
                {s}
              </button>
            ))}
          </div>
        )}

        {i && i.explicacion && (
          <p className="flex items-center gap-1.5 text-[11px] text-slate-500 pt-1 border-t border-slate-700/60">
            {i.motor === 'llm' ? (
              <span className="inline-flex items-center gap-1 text-violet-300 bg-violet-500/10 rounded px-1.5 py-0.5" title="La pregunta la ha interpretado el modelo de lenguaje local">
                <Cpu className="w-3 h-3" /> IA local
              </span>
            ) : (
              <span className="inline-flex items-center gap-1 text-slate-400 bg-slate-700/40 rounded px-1.5 py-0.5" title="La pregunta la ha interpretado el motor de reglas">
                <ListChecks className="w-3 h-3" /> reglas
              </span>
            )}
            <span>Entendido: {i.explicacion}</span>
          </p>
        )}
      </div>
    </div>
  );
}

export default function ChatAsistente() {
  const [abierto, setAbierto] = useState(false);
  const [mensajes, setMensajes] = useState([
    { rol: 'asistente', texto: '¡Hola! Pregúntame lo que quieras sobre tus gastos. Por ejemplo:', sugerencias: SUGERENCIAS_INICIALES },
  ]);
  const [texto, setTexto] = useState('');
  const [cargando, setCargando] = useState(false);
  // Motor del asistente: 'llm' (IA local + reglas de respaldo) o 'reglas' (solo motor propio, sin LLM)
  const [motor, setMotor] = useState(() => {
    try { return localStorage.getItem('motorAsistente') || 'llm'; } catch { return 'llm'; }
  });
  const [estadoLLM, setEstadoLLM] = useState(null);   // respuesta de /chat/estado
  const finRef = useRef(null);
  const inputRef = useRef(null);

  useEffect(() => {
    finRef.current?.scrollIntoView({ behavior: 'smooth' });
  }, [mensajes, cargando]);

  useEffect(() => {
    if (abierto) inputRef.current?.focus();
  }, [abierto]);

  // Al abrir el chat se comprueba si el LLM local está en marcha
  useEffect(() => {
    if (!abierto) return;
    axios.get(`${API}/chat/estado`)
      .then(({ data }) => setEstadoLLM(data))
      .catch(() => setEstadoLLM({ conectado: false }));
  }, [abierto]);

  const cambiarMotor = (nuevo) => {
    setMotor(nuevo);
    try { localStorage.setItem('motorAsistente', nuevo); } catch { /* sin almacenamiento: solo esta sesión */ }
  };
  const llmListo = estadoLLM?.activado !== false && estadoLLM?.conectado && estadoLLM?.modelo_instalado !== false;

  const enviar = async (pregunta) => {
    const mensaje = (pregunta ?? texto).trim();
    if (!mensaje || cargando) return;
    setTexto('');
    setMensajes((prev) => [...prev, { rol: 'usuario', texto: mensaje }]);
    setCargando(true);
    try {
      // Últimas preguntas con su consulta interpretada, para entender preguntas como "¿y el mes pasado?"
      const historial = [];
      mensajes.forEach((m, idx) => {
        const anterior = mensajes[idx - 1];
        if (m.rol === 'asistente' && m.interpretacion?.consulta && anterior?.rol === 'usuario') {
          historial.push({ pregunta: anterior.texto, consulta: m.interpretacion.consulta });
        }
      });
      const { data } = await axios.post(`${API}/chat`, { mensaje, historial: historial.slice(-3), motor });
      setMensajes((prev) => [...prev, {
        rol: 'asistente', texto: data.respuesta, tabla: data.tabla, nota: data.nota,
        sugerencias: data.sugerencias, interpretacion: data.interpretacion,
      }]);
    } catch (e) {
      console.error('Error en el chat:', e);
      setMensajes((prev) => [...prev, { rol: 'asistente', texto: 'No he podido conectar con el servidor. Inténtalo de nuevo.' }]);
    } finally {
      setCargando(false);
    }
  };

  return (
    <>
      {!abierto && (
        <button
          onClick={() => setAbierto(true)}
          className="fixed bottom-6 right-6 z-40 flex items-center gap-2 bg-blue-600 hover:bg-blue-500 text-white rounded-full pl-4 pr-5 py-3 shadow-lg shadow-blue-900/40 transition-colors"
          aria-label="Abrir asistente de gastos"
        >
          <MessageCircle className="w-5 h-5" />
          <span className="font-medium text-sm">Pregunta por tus gastos</span>
        </button>
      )}

      {abierto && (
        <div className="fixed z-40 inset-0 sm:inset-auto sm:bottom-6 sm:right-6 sm:w-[440px] sm:h-[640px] sm:max-h-[calc(100vh-3rem)] flex flex-col bg-slate-900 border border-slate-700 sm:rounded-2xl shadow-2xl">
          <div className="flex items-center justify-between px-4 py-3 border-b border-slate-800">
            <div className="flex items-center gap-2">
              <div className="w-8 h-8 rounded-full bg-blue-600/20 flex items-center justify-center">
                <Sparkles className="w-4 h-4 text-blue-400" />
              </div>
              <div>
                <p className="text-sm font-semibold text-white">Asistente de gastos</p>
              </div>
            </div>
            <div className="flex items-center gap-1">
            <div className="flex items-center bg-slate-950 border border-slate-700 rounded-lg p-0.5 text-[11px]" role="radiogroup" aria-label="Motor del asistente">
              <button
                role="radio" aria-checked={motor === 'llm'}
                onClick={() => cambiarMotor('llm')}
                title={estadoLLM && !llmListo ? 'El LLM local no responde: se usará el motor de reglas' : 'Un LLM local interpreta la pregunta; los cálculos los hace el motor propio'}
                className={`inline-flex items-center gap-1 whitespace-nowrap px-2 py-1 rounded-md transition-colors ${motor === 'llm' ? 'bg-violet-600 text-white' : 'text-slate-400 hover:text-white'}`}
              >
                <Cpu className="w-3 h-3" /> IA local
                {motor === 'llm' && estadoLLM && (
                  <span className={`w-1.5 h-1.5 rounded-full ${llmListo ? 'bg-emerald-400' : 'bg-red-400'}`} />
                )}
              </button>
              <button
                role="radio" aria-checked={motor === 'reglas'}
                onClick={() => cambiarMotor('reglas')}
                title="Solo el motor propio (clasificador de intenciones + reglas), sin LLM"
                className={`inline-flex items-center gap-1 whitespace-nowrap px-2 py-1 rounded-md transition-colors ${motor === 'reglas' ? 'bg-slate-600 text-white' : 'text-slate-400 hover:text-white'}`}
              >
                <ListChecks className="w-3 h-3" /> Sin LLM
              </button>
            </div>
            <button onClick={() => setAbierto(false)} className="p-1.5 rounded-lg text-slate-400 hover:text-white hover:bg-slate-800" aria-label="Cerrar">
              <X className="w-5 h-5" />
            </button>
            </div>
          </div>
          {motor === 'llm' && estadoLLM && !llmListo && (
            <p className="px-4 py-2 text-[11px] text-amber-300 bg-amber-500/10 border-b border-slate-800">
              {estadoLLM.conectado
                ? `El modelo «${estadoLLM.modelo}» no está instalado en el servidor LLM.`
                : 'No se puede conectar con el LLM local (¿está abierto Ollama?).'} Mientras tanto responde el motor de reglas.
            </p>
          )}

          <div className="flex-1 overflow-y-auto px-4 py-4 space-y-3">
            {mensajes.map((m, i) => <Mensaje key={i} m={m} onSugerencia={enviar} />)}
            {cargando && (
              <div className="flex items-center gap-2 text-slate-400 text-sm">
                <Loader2 className="w-4 h-4 animate-spin" /> Calculando…
              </div>
            )}
            <div ref={finRef} />
          </div>

          <form
            onSubmit={(e) => { e.preventDefault(); enviar(); }}
            className="flex items-center gap-2 p-3 border-t border-slate-800"
          >
            <input
              ref={inputRef}
              value={texto}
              onChange={(e) => setTexto(e.target.value)}
              placeholder="Ej.: ¿Cuánto gasto en carne al mes?"
              className="flex-1 bg-slate-950 border border-slate-700 rounded-xl px-3 py-2.5 text-sm text-white placeholder-slate-600 focus:outline-none focus:border-blue-500"
            />
            <button
              type="submit"
              disabled={!texto.trim() || cargando}
              className="p-2.5 rounded-xl bg-blue-600 hover:bg-blue-500 text-white disabled:opacity-40"
              aria-label="Enviar"
            >
              <Send className="w-4 h-4" />
            </button>
          </form>
        </div>
      )}
    </>
  );
}
