import { Loader2, CheckCircle, Copy, AlertTriangle, XCircle, Clock, X, Eye, Ban } from 'lucide-react';

// Estado de cada archivo de una subida múltiple
const ESTADOS = {
  pendiente:  { texto: 'En cola',         icono: Clock,         clase: 'text-slate-500' },
  procesando: { texto: 'Procesando…',     icono: Loader2,       clase: 'text-blue-400', girar: true },
  guardado:   { texto: 'Guardado',        icono: CheckCircle,   clase: 'text-emerald-400' },
  repetido:   { texto: 'Repetido',        icono: Copy,          clase: 'text-amber-400' },
  revisar:    { texto: 'Revisar',         icono: AlertTriangle, clase: 'text-orange-400' },
  error:      { texto: 'Error',           icono: XCircle,       clase: 'text-red-400' },
  descartado: { texto: 'Descartado',      icono: Ban,           clase: 'text-slate-500' },
  cancelado:  { texto: 'No procesado',    icono: Ban,           clase: 'text-slate-500' },
};

export default function PanelLote({ lote, activo, onCancelar, onRevisar, onDescartar, onCerrar }) {
  if (!lote.length) return null;
  const cuenta = (e) => lote.filter((x) => x.estado === e).length;
  const hechos = lote.filter((x) => !['pendiente', 'procesando'].includes(x.estado)).length;
  const porcentaje = Math.round((hechos / lote.length) * 100);

  return (
    <div className="mt-4 bg-slate-900 border border-slate-800 rounded-xl p-5">
      <div className="flex flex-wrap items-center justify-between gap-3 mb-3">
        <div>
          <h3 className="text-white font-semibold">Subida múltiple · {hechos}/{lote.length} archivos</h3>
          <p className="text-xs text-slate-400 mt-0.5">
            {cuenta('guardado')} guardados · {cuenta('repetido')} repetidos · {cuenta('revisar')} por revisar
            {cuenta('error') > 0 && ` · ${cuenta('error')} con error`}
          </p>
        </div>
        {activo ? (
          <button onClick={onCancelar}
            className="px-3 py-1.5 text-sm rounded-lg border border-slate-700 text-slate-300 hover:bg-slate-800">
            Detener después de este
          </button>
        ) : (
          <button onClick={onCerrar} aria-label="Cerrar resumen"
            className="p-1.5 rounded-lg text-slate-400 hover:text-white hover:bg-slate-800">
            <X className="w-5 h-5" />
          </button>
        )}
      </div>

      <div className="h-1.5 bg-slate-800 rounded-full overflow-hidden mb-4">
        <div className="h-full bg-blue-500 transition-all" style={{ width: `${porcentaje}%` }} />
      </div>

      <ul className="divide-y divide-slate-800 max-h-80 overflow-y-auto">
        {lote.map((x) => {
          const e = ESTADOS[x.estado] || ESTADOS.pendiente;
          const Icono = e.icono;
          return (
            <li key={x.id} className="flex items-center gap-3 py-2 text-sm">
              <Icono className={`w-4 h-4 shrink-0 ${e.clase} ${e.girar ? 'animate-spin' : ''}`} />
              <div className="min-w-0 flex-1">
                <p className="text-slate-200 truncate">{x.nombre}</p>
                {x.mensaje && <p className="text-xs text-slate-500 truncate" title={x.mensaje}>{x.mensaje}</p>}
              </div>
              {x.estado !== 'revisar' && <span className={`text-xs ${e.clase} shrink-0`}>{e.texto}</span>}
              {x.estado === 'revisar' && (
                <div className="flex gap-1 shrink-0">
                  <button onClick={() => onRevisar(x)}
                    className="inline-flex items-center gap-1 px-2.5 py-1 text-xs rounded-md bg-blue-600 hover:bg-blue-500 text-white">
                    <Eye className="w-3.5 h-3.5" /> Revisar
                  </button>
                  <button onClick={() => onDescartar(x)} aria-label="Descartar"
                    className="px-2 py-1 text-xs rounded-md border border-slate-700 text-slate-400 hover:text-white hover:bg-slate-800">
                    Descartar
                  </button>
                </div>
              )}
            </li>
          );
        })}
      </ul>
    </div>
  );
}
