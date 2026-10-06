import { useState, useEffect, useCallback, useRef } from 'react';
import axios from 'axios';
import { UploadCloud, Loader2, CheckCircle, Plus, Trash2, X, AlertTriangle, CalendarDays, Search, ArrowDownWideNarrow, ArrowUpNarrowWide } from 'lucide-react';
import { toast, Toaster } from 'sonner';
import ChatAsistente from './components/ChatAsistente';
import PanelLote from './components/PanelLote';

const API = `${process.env.REACT_APP_BACKEND_URL}/api`;

export default function App() {
  const [isDragging, setIsDragging] = useState(false);
  const [isUploading, setIsUploading] = useState(false);
  const [tickets, setTickets] = useState([]);
  const [loadingDatos, setLoadingDatos] = useState(true);

  // --- NUEVOS ESTADOS PARA LA REVISIÓN HUMANA ---
  const [draftTicket, setDraftTicket] = useState(null);
  const [isSaving, setIsSaving] = useState(false);

  // --- CATEGORÍAS (catálogo de Mercadona) ---
  const [arbolCategorias, setArbolCategorias] = useState([]);
  const [filtroCategoria, setFiltroCategoria] = useState('');
  const [filtroSubcategoria, setFiltroSubcategoria] = useState('');
  const [filtroMes, setFiltroMes] = useState('');      // "YYYY-MM"
  const [filtroDia, setFiltroDia] = useState('');      // "YYYY-MM-DD"
  const [busqueda, setBusqueda] = useState('');        // búsqueda por nombre del ticket
  const [ordenCampo, setOrdenCampo] = useState('fecha');   // 'fecha' | 'precio'
  const [ordenDir, setOrdenDir] = useState('desc');        // 'desc' (mayor a menor) | 'asc'
  const [confirmarBorrado, setConfirmarBorrado] = useState(null);   // id del ticket pendiente de confirmar
  const [borrando, setBorrando] = useState(null);

  // --- SUBIDA MÚLTIPLE ---
  const [lote, setLote] = useState([]);              // [{id, nombre, estado, mensaje, borrador}]
  const [loteActivo, setLoteActivo] = useState(false);
  const [revisarTodos, setRevisarTodos] = useState(false);
  const cancelarLote = useRef(false);

  useEffect(() => {
    axios.get(`${API}/categorias`)
      .then((res) => setArbolCategorias(res.data))
      .catch((e) => console.error('Error cargando categorías:', e));
  }, []);

  // Cargar los tickets ya procesados en la Base de Datos
  const fetchTickets = useCallback(async () => {
    setLoadingDatos(true);
    try {
      const res = await axios.get(`${API}/tickets`);
      setTickets(res.data);
    } catch (e) {
      console.error("Error cargando tickets:", e);
      toast.error("Error al cargar los datos procesados");
    } finally {
      setLoadingDatos(false);
    }
  }, []);

  useEffect(() => {
    fetchTickets();
  }, [fetchTickets]);

  // ── Subida de tickets ──

  // Envía un archivo al backend junto con su fecha de modificación (sirve para comprobar la fecha leída)
  const subirArchivo = async (file) => {
    const formData = new FormData();
    formData.append('file', file);
    if (file.lastModified) formData.append('ultima_modificacion', String(file.lastModified));
    const res = await axios.post(`${API}/upload`, formData);
    return res.data;
  };

  const borradorDesdeRespuesta = (data) => ({
    nombre_personalizado: '',
    tienda: data.tienda || 'Desconocida',
    fecha_compra: data.fecha_compra || '',
    fecha_leida: data.fecha_compra || '',
    items: (data.items || []).map((it) => ({ ...it, categorias: it.categorias || [] })),
    precio_total: data.precio_total || 0,
    upload_id: data.upload_id || data.id,
    archivo_hash: data.archivo_hash,
    revision_fecha: data.revision_fecha || null,
  });

  // Explica por qué el backend no ha devuelto productos (error del modelo, extracto bancario, formato…)
  const motivoSinDatos = (data) => {
    if (data?.result_type === 'banco') return `Se ha interpretado como extracto bancario (${data.result_count} movimientos), no como ticket`;
    if (data?.result_type === 'formato_no_soportado') return 'Formato de archivo no soportado';
    if (data?.status === 'error' && data?.result_type) return `Error al procesar: ${data.result_type}`;
    return 'No se pudieron extraer los datos del ticket';
  };

  // Lo que se manda a /save-ticket (sin los campos que solo usa la interfaz)
  const ticketParaGuardar = ({ revision_fecha, fecha_leida, _loteId, ...resto }) => resto;

  // Motivos por los que un ticket de una subida múltiple no se guarda solo y pasa a revisión
  const motivosRevision = (b) => {
    const motivos = [];
    if (b.revision_fecha?.aviso) motivos.push('fecha dudosa');
    if (!b.items.length) motivos.push('sin productos');
    if (b.items.some((it) => !String(it.descripcion || '').trim())) motivos.push('productos sin nombre');
    const suma = b.items.reduce((acc, it) => acc + (Number(it.precio_unitario) || 0) * (Number(it.cantidad) || 1), 0);
    if (Math.abs(suma - b.precio_total) > 0.01) motivos.push('el total no cuadra con los productos');
    return motivos;
  };

  const handleUpload = async (files) => {
    if (!files.length || isUploading || loteActivo) return;
    if (files.length > 1) {
      procesarLote(files);
      return;
    }
    setIsUploading(true);
    const file = files[0];
    try {
      const data = await subirArchivo(file);

      // Ticket repetido: se avisa y no se abre la revisión
      if (data?.duplicado) {
        toast.warning('Ticket repetido', { description: data.mensaje, duration: 8000 });
        return;
      }
      if (data && data.items) {
        const borrador = borradorDesdeRespuesta(data);
        if (borrador.revision_fecha?.aviso) {
          toast.warning('Revisa la fecha del ticket', { description: borrador.revision_fecha.aviso, duration: 8000 });
        } else {
          toast.success('IA procesada. Por favor, revisa los datos.');
        }
        setDraftTicket(borrador);
      } else {
        toast.error('No se ha podido leer el ticket', { description: motivoSinDatos(data), duration: 10000 });
      }
    } catch (e) {
      console.error(`Error subiendo ${file.name}:`, e);
      toast.error('Error al comunicarse con la IA.');
    } finally {
      setIsUploading(false);
    }
  };

  // Subida múltiple: los archivos se procesan uno a uno. Los tickets sin problemas se guardan solos
  // (el backend sigue comprobando duplicados con la base de datos, incluidos los guardados en este mismo lote);
  // los dudosos quedan en la lista para revisarlos.
  const actualizarLote = (id, cambios) => setLote((prev) => prev.map((x) => (x.id === id ? { ...x, ...cambios } : x)));

  const procesarLote = async (files) => {
    const entradas = files.map((f, i) => ({ id: `${Date.now()}-${i}`, nombre: f.name, estado: 'pendiente', mensaje: '' }));
    setLote(entradas);
    setLoteActivo(true);
    cancelarLote.current = false;
    let guardados = 0, repetidos = 0, porRevisar = 0, errores = 0;

    for (let i = 0; i < files.length; i++) {
      const { id } = entradas[i];
      if (cancelarLote.current) {
        actualizarLote(id, { estado: 'cancelado', mensaje: 'Subida detenida' });
        continue;
      }
      actualizarLote(id, { estado: 'procesando' });
      try {
        const data = await subirArchivo(files[i]);
        if (data?.duplicado) {
          repetidos++;
          actualizarLote(id, { estado: 'repetido', mensaje: data.mensaje });
          continue;
        }
        if (!data?.items) {
          errores++;
          actualizarLote(id, { estado: 'error', mensaje: motivoSinDatos(data) });
          continue;
        }
        const borrador = borradorDesdeRespuesta(data);
        const motivos = revisarTodos ? ['revisión manual activada'] : motivosRevision(borrador);
        const resumen = `${borrador.fecha_compra} · ${Number(borrador.precio_total).toFixed(2)} € · ${borrador.items.length} productos`;
        if (motivos.length) {
          porRevisar++;
          actualizarLote(id, { estado: 'revisar', mensaje: `${resumen} — ${motivos.join(', ')}`, borrador });
          continue;
        }
        try {
          await axios.post(`${API}/save-ticket`, ticketParaGuardar(borrador));
          guardados++;
          actualizarLote(id, { estado: 'guardado', mensaje: resumen });
        } catch (e) {
          if (e.response?.status !== 409) throw e;
          repetidos++;
          actualizarLote(id, { estado: 'repetido', mensaje: e.response.data?.detail });
        }
      } catch (e) {
        console.error(`Error procesando ${files[i].name}:`, e);
        errores++;
        actualizarLote(id, { estado: 'error', mensaje: `Error al procesar el archivo${e.response ? ` (HTTP ${e.response.status})` : ' (sin respuesta del servidor)'}` });
      }
    }

    setLoteActivo(false);
    fetchTickets();
    toast.success('Subida múltiple terminada', {
      description: `${guardados} guardados, ${repetidos} repetidos omitidos, ${porRevisar} por revisar${errores ? `, ${errores} con error` : ''}.`,
      duration: 8000,
    });
  };

  const revisarDeLote = (x) => setDraftTicket({ ...x.borrador, _loteId: x.id });
  const descartarDeLote = (x) => actualizarLote(x.id, { estado: 'descartado', borrador: null });

  // Cambiar la fecha en la revisión conservando la hora, si la había ("dd/mm/yyyy HH:MM")
  const handleFechaChange = (iso) => {
    if (!iso) return;
    const [y, m, d] = iso.split('-');
    setDraftTicket((prev) => {
      const hora = (/\s(\d{1,2}:\d{2})/.exec(prev.fecha_compra || '') || [])[1];
      return { ...prev, fecha_compra: `${d}/${m}/${y}${hora ? ' ' + hora : ''}` };
    });
  };

  // --- MANEJADORES DE ESTADO (REVISIÓN HUMANA) ---

  const handleNameChange = (e) => {
    setDraftTicket((prev) => ({
      ...prev,
      nombre_personalizado: e.target.value
    }));
  };

  const handleItemChange = (index, field, value) => {
    setDraftTicket((prev) => {
      const newItems = [...prev.items];
      newItems[index] = { ...newItems[index], [field]: value };
      return { ...prev, items: newItems };
    });
  };

  // Añadir un producto que la IA no ha leído
  const handleAddItem = () => {
    setDraftTicket((prev) => ({
      ...prev,
      items: [
        ...prev.items,
        { descripcion: '', cantidad: 1, precio_unitario: 0, categorias: [], añadido_manual: true, categoria_manual: false },
      ],
    }));
  };

  // Eliminar un producto leído por error
  const handleRemoveItem = (index) => {
    setDraftTicket((prev) => ({ ...prev, items: prev.items.filter((_, i) => i !== index) }));
  };

  // Al terminar de editar la descripción se vuelve a buscar en el catálogo,
  // salvo que el usuario ya haya elegido las categorías a mano.
  const handleDescripcionBlur = async (index) => {
    const item = draftTicket.items[index];
    if (!item || item.categoria_manual || !item.descripcion.trim()) return;
    try {
      const res = await axios.post(`${API}/categorizar`, { descripcion: item.descripcion });
      setDraftTicket((prev) => {
        const newItems = [...prev.items];
        if (newItems[index]?.descripcion !== item.descripcion) return prev; // cambió mientras tanto
        newItems[index] = {
          ...newItems[index],
          categorias: res.data.categorias || [],
          producto_catalogo: res.data.producto_catalogo,
          origen_categoria: res.data.origen,
        };
        return { ...prev, items: newItems };
      });
    } catch (e) {
      console.error('Error categorizando:', e);
    }
  };

  // Cambia el nombre de un producto (aceptar sugerencia / deshacer corrección) y vuelve a categorizarlo
  const cambiarNombreItem = async (index, descripcion, extra = {}) => {
    setDraftTicket((prev) => {
      const newItems = [...prev.items];
      newItems[index] = { ...newItems[index], descripcion, sugerencia_nombre: null, ...extra };
      return { ...prev, items: newItems };
    });
    if (draftTicket.items[index]?.categoria_manual) return;
    try {
      const res = await axios.post(`${API}/categorizar`, { descripcion });
      setDraftTicket((prev) => {
        const newItems = [...prev.items];
        if (newItems[index]?.descripcion !== descripcion) return prev;
        newItems[index] = { ...newItems[index], categorias: res.data.categorias || [],
          producto_catalogo: res.data.producto_catalogo, origen_categoria: res.data.origen };
        return { ...prev, items: newItems };
      });
    } catch (e) {
      console.error('Error categorizando:', e);
    }
  };

  const handleAddCategoria = (index, valor) => {
    if (!valor) return;
    const [categoria, subcategoria] = valor.split('||');
    setDraftTicket((prev) => {
      const newItems = [...prev.items];
      const actuales = newItems[index].categorias || [];
      if (actuales.some((c) => c.categoria === categoria && c.subcategoria === subcategoria)) return prev;
      newItems[index] = {
        ...newItems[index],
        categorias: [...actuales, { categoria, subcategoria }],
        categoria_manual: true,
      };
      return { ...prev, items: newItems };
    });
  };

  const handleRemoveCategoria = (index, catIndex) => {
    setDraftTicket((prev) => {
      const newItems = [...prev.items];
      newItems[index] = {
        ...newItems[index],
        categorias: newItems[index].categorias.filter((_, i) => i !== catIndex),
        categoria_manual: true,
      };
      return { ...prev, items: newItems };
    });
  };

  const handleTotalChange = (e) => {
    const valor = parseFloat(e.target.value);
    setDraftTicket((prev) => ({ ...prev, precio_total: isNaN(valor) ? 0 : valor }));
  };

  const sumaProductos = draftTicket
    ? draftTicket.items.reduce((acc, it) => acc + (Number(it.precio_unitario) || 0) * (Number(it.cantidad) || 1), 0)
    : 0;
  const totalDescuadrado = draftTicket && Math.abs(sumaProductos - draftTicket.precio_total) > 0.01;

  // ── Filtros del historial ──
  // fecha_compra llega como "dd/mm/yyyy" (o "dd/mm/yyyy HH:MM"); se pasa a "YYYY-MM-DD" para ordenar y filtrar
  const fechaISO = (fecha) => {
    const m = /^(\d{1,2})\/(\d{1,2})\/(\d{4})/.exec(String(fecha || '').trim());
    if (m) return `${m[3]}-${m[2].padStart(2, '0')}-${m[1].padStart(2, '0')}`;
    const d = new Date(fecha);
    return isNaN(d) ? '' : d.toISOString().slice(0, 10);
  };

  const NOMBRES_MES = ['enero', 'febrero', 'marzo', 'abril', 'mayo', 'junio', 'julio', 'agosto',
    'septiembre', 'octubre', 'noviembre', 'diciembre'];
  const nombreMes = (yyyymm) => {
    const [y, m] = yyyymm.split('-');
    return `${NOMBRES_MES[Number(m) - 1]} ${y}`;
  };

  // Meses que tienen algún ticket, del más reciente al más antiguo
  const mesesDisponibles = [...new Set(tickets.map((t) => fechaISO(t.fecha_compra).slice(0, 7)).filter(Boolean))]
    .sort().reverse();

  const subcategoriasDisponibles = arbolCategorias.find((g) => g.categoria === filtroCategoria)?.subcategorias || [];

  const productoEnFiltro = (item) => {
    if (!filtroCategoria) return true;
    return (item.categorias || []).some((c) =>
      c.categoria === filtroCategoria && (!filtroSubcategoria || c.subcategoria === filtroSubcategoria));
  };

  const importeItem = (it) => (Number(it.precio_unitario) || 0) * (Number(it.cantidad) || 1);

  // Búsqueda sin distinguir mayúsculas ni tildes ("compra semanal" encuentra "Compra Semanal")
  const sinTildes = (txt) => String(txt || '').normalize('NFD').replace(/[\u0300-\u036f]/g, '').toLowerCase().trim();
  const nombreTicket = (t) => t.nombre_personalizado || t.tienda || '';
  const textoBuscado = sinTildes(busqueda);
  const horaTicket = (t) => (/\s(\d{1,2}):(\d{2})/.exec(t.fecha_compra || '') || []).slice(1).map((x) => x.padStart(2, '0')).join(':');

  const ticketsFiltrados = tickets
    .map((t) => ({ ...t, _fecha: fechaISO(t.fecha_compra) }))
    .filter((t) => !filtroMes || t._fecha.startsWith(filtroMes))
    .filter((t) => !filtroDia || t._fecha === filtroDia)
    .filter((t) => !textoBuscado || sinTildes(nombreTicket(t)).includes(textoBuscado))
    .map((t) => ({ ...t, _items: (t.items || []).filter(productoEnFiltro) }))
    .filter((t) => !filtroCategoria || t._items.length > 0)
    // Importe que se muestra en la tarjeta: con filtro de categoría, solo lo de esa categoría
    .map((t) => ({ ...t, _importe: filtroCategoria ? t._items.reduce((s, it) => s + importeItem(it), 0) : Number(t.precio_total) || 0,
                       _orden: `${t._fecha} ${horaTicket(t)}` }))
    .sort((a, b) => {
      const signo = ordenDir === 'asc' ? 1 : -1;
      const porFecha = a._orden.localeCompare(b._orden);
      if (ordenCampo === 'precio') return signo * ((a._importe - b._importe) || porFecha);
      return signo * porFecha;
    });
  const gastoFiltrado = ticketsFiltrados.reduce((acc, t) =>
    acc + (filtroCategoria ? t._items.reduce((s, it) => s + importeItem(it), 0) : Number(t.precio_total) || 0), 0);

  const hayFiltros = filtroCategoria || filtroMes || filtroDia || busqueda;
  const limpiarFiltros = () => {
    setFiltroCategoria(''); setFiltroSubcategoria(''); setFiltroMes(''); setFiltroDia(''); setBusqueda('');
  };
  const formatoFecha = (iso) => iso ? iso.split('-').reverse().join('/') : '';
  const claseSelect = 'bg-slate-900 border border-slate-700 rounded-lg px-3 py-2 text-sm text-slate-300 focus:outline-none focus:border-blue-500';

  const handleDeleteTicket = async (ticket) => {
    setBorrando(ticket.id);
    try {
      await axios.delete(`${API}/tickets/${ticket.id}`);
      setTickets((prev) => prev.filter((t) => t.id !== ticket.id));
      toast.success('Ticket eliminado', {
        description: `${ticket.nombre_personalizado || ticket.tienda} del ${ticket.fecha_compra}`,
      });
    } catch (e) {
      console.error('Error eliminando el ticket:', e);
      toast.error('No se pudo eliminar el ticket.');
    } finally {
      setBorrando(null);
      setConfirmarBorrado(null);
    }
  };

  const handleSaveTicket = async () => {
    if (draftTicket.items.some((it) => !it.descripcion.trim())) {
      toast.error('Hay productos sin descripción. Rellénalos o elimínalos antes de guardar.');
      return;
    }
    setIsSaving(true);
    try {
      // Enviamos el ticket definitivo al nuevo endpoint del backend
      const response = await axios.post(`${API}/save-ticket`, ticketParaGuardar(draftTicket));
      
      if (response.status === 200 || response.status === 201) {
        toast.success("¡Ticket confirmado y guardado con éxito!");
        if (draftTicket._loteId) {
          actualizarLote(draftTicket._loteId, {
            estado: 'guardado', borrador: null,
            mensaje: `${draftTicket.fecha_compra} · ${Number(draftTicket.precio_total).toFixed(2)} € · revisado`,
          });
        }
        setDraftTicket(null); // Oculta la pantalla de revisión
        fetchTickets(); // Refresca la tabla inferior
      }
    } catch (error) {
      if (error.response?.status === 409) {
        // El ticket revisado coincide exactamente con uno ya guardado
        toast.warning('Ticket repetido', { description: error.response.data?.detail, duration: 8000 });
        if (draftTicket._loteId) {
          actualizarLote(draftTicket._loteId, { estado: 'repetido', borrador: null, mensaje: error.response.data?.detail });
        }
        setDraftTicket(null);
        return;
      }
      console.error("Error al guardar el ticket revisado:", error);
      toast.error("Hubo un error al conectar con la base de datos.");
    } finally {
      setIsSaving(false);
    }
  };

  // Manejadores de Drag & Drop
  const onDragOver = (e) => { e.preventDefault(); e.stopPropagation(); setIsDragging(true); };
  const onDragLeave = (e) => { e.preventDefault(); e.stopPropagation(); setIsDragging(false); };
  const onDrop = (e) => {
    e.preventDefault();
    setIsDragging(false);
    const validFiles = Array.from(e.dataTransfer.files).filter(file => 
      ['image/jpeg', 'image/png', 'image/webp', 'application/pdf'].includes(file.type)
    );
    handleUpload(validFiles);
  };

  return (
    <div className="min-h-screen bg-slate-950 text-slate-200 font-sans p-6 md:p-12">
      <Toaster theme="dark" richColors position="top-center" closeButton />
      {!draftTicket && <ChatAsistente />}
      <div className="max-w-6xl mx-auto space-y-10">
        
        {/* Cabecera */}
        <header className="border-b border-slate-800 pb-6">
          <h1 className="text-3xl font-bold text-white">FinTrack Pro</h1>
          <p className="text-slate-400 mt-2">Sube tus tickets y verifica los datos extraídos por la IA.</p>
        </header>

        {/* 
            RENDERIZADO CONDICIONAL: 
            Si draftTicket tiene datos -> Mostramos pantalla de revisión
            Si draftTicket es null -> Mostramos zona de subida (Dropzone)
        */}
        {!draftTicket ? (
          <section>
            <div
              className={`border-2 border-dashed rounded-xl p-10 flex flex-col items-center justify-center cursor-pointer transition-all ${
                isDragging ? 'border-blue-500 bg-blue-500/10' : 'border-slate-700 bg-slate-900 hover:border-slate-500'
              }`}
              onDragOver={onDragOver}
              onDragLeave={onDragLeave}
              onDrop={onDrop}
              onClick={() => !isUploading && !loteActivo && document.getElementById('file-input').click()}
            >
              {isUploading || loteActivo ? (
                <Loader2 className="w-12 h-12 text-blue-500 animate-spin mb-4" />
              ) : (
                <UploadCloud className="w-12 h-12 text-blue-500 mb-4" />
              )}
              <p className="text-lg font-medium text-white">
                {isUploading ? 'La IA está procesando el ticket...'
                  : loteActivo ? 'Procesando la subida múltiple...'
                  : 'Arrastra tus tickets (PDF, JPG, PNG) aquí'}
              </p>
              <p className="text-sm text-slate-500 mt-2">
                O haz clic para seleccionarlos. Puedes subir uno o muchos a la vez; los repetidos se omiten.
              </p>
              <input
                id="file-input"
                type="file"
                multiple
                accept=".jpg,.jpeg,.png,.webp,.pdf"
                className="hidden"
                onChange={(e) => {
                  handleUpload(Array.from(e.target.files));
                  e.target.value = null; // Evita el problema del "Archivo Fantasma"
                }}
              />
            </div>
            <label className="mt-3 inline-flex items-center gap-2 text-sm text-slate-400 cursor-pointer select-none">
              <input type="checkbox" checked={revisarTodos} onChange={(e) => setRevisarTodos(e.target.checked)}
                className="accent-blue-500" disabled={loteActivo} />
              En subidas múltiples, revisar todos los tickets a mano (si no, solo se piden revisar los dudosos)
            </label>
            <PanelLote
              lote={lote}
              activo={loteActivo}
              onCancelar={() => { cancelarLote.current = true; }}
              onRevisar={revisarDeLote}
              onDescartar={descartarDeLote}
              onCerrar={() => setLote([])}
            />
          </section>
        ) : (
          <section className="bg-slate-900 border border-slate-800 rounded-xl p-6 md:p-8 shadow-xl animate-fade-up">
            {/* Cabecera de la revisión */}
            <div className="flex flex-col md:flex-row justify-between items-start md:items-center mb-6 border-b border-slate-800 pb-6 gap-6">
              <div className="w-full max-w-md">
                <h2 className="text-2xl font-bold text-white mb-2">Revisión del Ticket</h2>
                <p className="text-sm text-slate-400 mb-4">Verifica los datos de la IA antes de guardarlos en la BD.</p>
                
                <label className="block text-sm font-medium text-slate-300 mb-1">Nombre Personalizado (Opcional)</label>
                <input
                  type="text"
                  placeholder="Ej: Compra semanal Mercadona..."
                  value={draftTicket.nombre_personalizado}
                  onChange={handleNameChange}
                  className="w-full bg-slate-950 border border-slate-700 rounded-lg px-4 py-2 text-white placeholder-slate-600 focus:outline-none focus:border-blue-500 focus:ring-1 focus:ring-blue-500 transition-all"
                />

                <label className="block text-sm font-medium text-slate-300 mt-4 mb-1">Fecha de compra</label>
                <div className="flex items-center gap-2">
                  <CalendarDays className="w-4 h-4 text-slate-500" />
                  <input
                    type="date"
                    value={fechaISO(draftTicket.fecha_compra)}
                    onChange={(e) => handleFechaChange(e.target.value)}
                    className="bg-slate-950 border border-slate-700 rounded-lg px-3 py-2 text-white focus:outline-none focus:border-blue-500 [color-scheme:dark]"
                  />
                  {/\s\d{1,2}:\d{2}/.test(draftTicket.fecha_compra) && (
                    <span className="text-xs text-slate-500">{draftTicket.fecha_compra.split(' ').slice(1).join(' ')}</span>
                  )}
                </div>
                {draftTicket.revision_fecha?.aviso && draftTicket.fecha_compra === draftTicket.fecha_leida && (
                  <div className="mt-2 p-3 rounded-lg border border-amber-500/30 bg-amber-500/10 text-xs text-amber-300 space-y-1.5">
                    <p className="flex items-start gap-2">
                      <AlertTriangle className="w-3.5 h-3.5 mt-0.5 shrink-0" />
                      <span>{draftTicket.revision_fecha.aviso}</span>
                    </p>
                    {draftTicket.revision_fecha.fecha_sugerida && (
                      <button
                        onClick={() => setDraftTicket((prev) => ({ ...prev, fecha_compra: prev.revision_fecha.fecha_sugerida }))}
                        className="ml-5 px-2.5 py-1 rounded-md bg-amber-500/20 hover:bg-amber-500/30 text-amber-200"
                      >
                        Usar {draftTicket.revision_fecha.fecha_sugerida.split(' ')[0]}
                        <span className="text-amber-400/70"> ({draftTicket.revision_fecha.motivo_sugerencia})</span>
                      </button>
                    )}
                  </div>
                )}
              </div>
              
              {/* Precio Total (editable) + comprobación con la suma de productos */}
              <div className="text-left md:text-right w-full md:w-auto bg-slate-950/50 p-4 rounded-lg border border-slate-800">
                <p className="text-sm font-medium text-slate-500 uppercase tracking-wider mb-1">Total del Ticket</p>
                <div className="flex items-center md:justify-end gap-1">
                  <input
                    type="number"
                    step="0.01"
                    value={draftTicket.precio_total}
                    onChange={handleTotalChange}
                    className="w-36 bg-transparent border border-transparent hover:border-slate-700 focus:border-blue-500 rounded px-2 py-1 text-4xl font-mono text-emerald-400 font-bold text-right focus:outline-none"
                  />
                  <span className="text-4xl font-mono text-emerald-400 font-bold">€</span>
                </div>
                <p className="text-xs text-slate-500 mt-2">Suma de productos: {sumaProductos.toFixed(2)} €</p>
                {totalDescuadrado && (
                  <div className="mt-2 flex items-center md:justify-end gap-2 text-xs text-amber-400">
                    <AlertTriangle className="w-3.5 h-3.5" />
                    <span>No cuadra con el total.</span>
                    <button
                      onClick={() => setDraftTicket((prev) => ({ ...prev, precio_total: Math.round(sumaProductos * 100) / 100 }))}
                      className="underline hover:text-amber-300"
                    >
                      Usar la suma
                    </button>
                  </div>
                )}
              </div>
            </div>

            {/* Tabla Interactiva */}
            <div className="overflow-x-auto mb-8 border border-slate-800 rounded-lg">
              <table className="w-full text-sm text-left">
                <thead className="text-xs text-slate-400 uppercase bg-slate-950">
                  <tr>
                    <th className="py-3 px-4 font-semibold">Descripción</th>
                    <th className="py-3 px-4 w-24 text-center font-semibold">Cant.</th>
                    <th className="py-3 px-4 w-32 text-right font-semibold">Precio/U (€)</th>
                    <th className="py-3 px-4 font-semibold">Categorías</th>
                    <th className="py-3 px-2 w-10"></th>
                  </tr>
                </thead>
                <tbody className="divide-y divide-slate-800/50">
                  {draftTicket.items.map((item, index) => (
                    <tr key={index} className="hover:bg-slate-800/30 transition-colors group">
                      <td className="py-2 px-2">
                        <input
                          type="text"
                          value={item.descripcion}
                          onChange={(e) => handleItemChange(index, 'descripcion', e.target.value)}
                          onBlur={() => handleDescripcionBlur(index)}
                          placeholder="Nombre del producto"
                          autoFocus={item.añadido_manual && !item.descripcion}
                          className="w-full bg-transparent border border-transparent group-hover:border-slate-700 focus:border-blue-500 rounded px-3 py-1.5 text-slate-200 placeholder-slate-600 focus:outline-none focus:bg-slate-950 transition-all"
                        />
                        {item.correccion_auto && item.descripcion !== item.correccion_auto.original && (
                          <p className="px-3 text-[11px] text-emerald-400/90 flex flex-wrap items-center gap-x-1.5"
                             title={`Corregido automáticamente: ${item.correccion_auto.motivo}`}>
                            <CheckCircle className="w-3 h-3 shrink-0" />
                            <span>Corregido · se leyó «{item.correccion_auto.original}»</span>
                            <button
                              onClick={() => cambiarNombreItem(index, item.correccion_auto.original, { correccion_auto: null })}
                              className="text-slate-400 hover:text-white underline decoration-dotted"
                            >
                              Deshacer
                            </button>
                          </p>
                        )}
                        {item.sugerencia_nombre && item.descripcion !== item.sugerencia_nombre && (
                          <button
                            onClick={() => cambiarNombreItem(index, item.sugerencia_nombre)}
                            className="px-3 text-[11px] text-amber-400 hover:text-amber-300 underline decoration-dotted text-left"
                            title="Producto parecido que ya has comprado antes"
                          >
                            ¿Quizá «{item.sugerencia_nombre}»?
                          </button>
                        )}
                        {item.producto_catalogo && !item.categoria_manual && (
                          <p className="px-3 text-[11px] text-slate-500 truncate" title={item.producto_catalogo}>
                            ≈ {item.producto_catalogo}
                          </p>
                        )}
                        {item.añadido_manual && (
                          <p className="px-3 text-[11px] text-blue-400">Añadido a mano</p>
                        )}
                      </td>
                      <td className="py-2 px-2">
                        <input
                          type="number"
                          min="1"
                          value={item.cantidad}
                          onChange={(e) => handleItemChange(index, 'cantidad', parseInt(e.target.value) || 1)}
                          className="w-full bg-transparent border border-transparent group-hover:border-slate-700 focus:border-blue-500 rounded px-3 py-1.5 text-center text-slate-200 focus:outline-none focus:bg-slate-950 transition-all"
                        />
                      </td>
                      <td className="py-2 px-2">
                        <input
                          type="number"
                          step="0.01"
                          value={item.precio_unitario}
                          onChange={(e) => handleItemChange(index, 'precio_unitario', parseFloat(e.target.value) || 0)}
                          className="w-full bg-transparent border border-transparent group-hover:border-slate-700 focus:border-blue-500 rounded px-3 py-1.5 text-right font-mono text-slate-200 focus:outline-none focus:bg-slate-950 transition-all"
                        />
                      </td>
                      <td className="py-2 px-2 min-w-[220px]">
                        <div className="flex flex-wrap gap-1 mb-1">
                          {(item.categorias || []).map((cat, ci) => (
                            <span
                              key={`${cat.categoria}-${cat.subcategoria}`}
                              title={cat.categoria}
                              className="inline-flex items-center gap-1 bg-blue-500/15 text-blue-300 border border-blue-500/30 rounded-full pl-2 pr-1 py-0.5 text-xs"
                            >
                              {cat.subcategoria || cat.categoria}
                              <button onClick={() => handleRemoveCategoria(index, ci)} className="hover:text-white" aria-label="Quitar categoría">
                                <X className="w-3 h-3" />
                              </button>
                            </span>
                          ))}
                          {(!item.categorias || item.categorias.length === 0) && (
                            <span className="text-xs text-amber-400/80">Sin categoría</span>
                          )}
                        </div>
                        <select
                          value=""
                          onChange={(e) => handleAddCategoria(index, e.target.value)}
                          className="w-full bg-slate-950 border border-slate-700 rounded px-2 py-1 text-xs text-slate-400 focus:outline-none focus:border-blue-500"
                        >
                          <option value="">+ Añadir categoría…</option>
                          {arbolCategorias.map((grupo) => (
                            <optgroup key={grupo.categoria} label={grupo.categoria}>
                              {grupo.subcategorias.map((sub) => (
                                <option key={sub} value={`${grupo.categoria}||${sub}`}>{sub}</option>
                              ))}
                            </optgroup>
                          ))}
                        </select>
                      </td>
                      <td className="py-2 px-2 text-center">
                        <button
                          onClick={() => handleRemoveItem(index)}
                          className="p-1.5 rounded text-slate-500 hover:text-red-400 hover:bg-red-500/10 transition-colors"
                          aria-label="Eliminar producto"
                        >
                          <Trash2 className="w-4 h-4" />
                        </button>
                      </td>
                    </tr>
                  ))}
                </tbody>
              </table>
            </div>

            <button
              onClick={handleAddItem}
              className="-mt-4 mb-8 inline-flex items-center gap-2 px-4 py-2 rounded-lg border border-dashed border-slate-700 text-slate-300 hover:border-blue-500 hover:text-white transition-colors text-sm"
            >
              <Plus className="w-4 h-4" /> Añadir producto
            </button>

            {/* Botones de Acción */}
            <div className="flex justify-end gap-3 pt-2">
              <button 
                onClick={() => setDraftTicket(null)}
                disabled={isSaving}
                className="px-5 py-2.5 rounded-lg border border-slate-700 text-slate-300 hover:bg-slate-800 hover:text-white transition-colors disabled:opacity-50 font-medium"
              >
                Descartar
              </button>
              <button 
                onClick={handleSaveTicket}
                disabled={isSaving}
                className="px-5 py-2.5 rounded-lg bg-blue-600 text-white font-medium hover:bg-blue-500 focus:ring-4 focus:ring-blue-500/20 transition-all flex items-center gap-2 disabled:opacity-50"
              >
                {isSaving ? (
                  <><Loader2 className="w-4 h-4 animate-spin" /> Guardando...</>
                ) : (
                  <><CheckCircle className="w-4 h-4" /> Confirmar y Guardar</>
                )}
              </button>
            </div>
          </section>
        )}

        {/* Resultados Extraídos (Oculto durante la revisión para evitar distracciones) */}
        {!draftTicket && (
          <section>
            <h2 className="text-xl font-semibold text-white mb-4">Historial de Tickets</h2>

            {/* Barra de filtros */}
            <div className="bg-slate-900 border border-slate-800 rounded-xl p-4 mb-6 space-y-4">
              {/* Búsqueda por nombre y orden */}
              <div className="flex flex-col md:flex-row gap-3">
                <div className="relative flex-1">
                  <Search className="w-4 h-4 text-slate-500 absolute left-3 top-1/2 -translate-y-1/2 pointer-events-none" />
                  <input
                    type="search"
                    value={busqueda}
                    onChange={(e) => setBusqueda(e.target.value)}
                    placeholder="Buscar ticket por nombre…"
                    aria-label="Buscar ticket por nombre"
                    className={`${claseSelect} w-full pl-9`}
                  />
                </div>
                <div className="flex items-center gap-2">
                  <label className="text-xs text-slate-400 whitespace-nowrap" htmlFor="orden-campo">Ordenar por</label>
                  <select id="orden-campo" value={ordenCampo} onChange={(e) => setOrdenCampo(e.target.value)} className={claseSelect}>
                    <option value="fecha">Fecha</option>
                    <option value="precio">Precio</option>
                  </select>
                  <button
                    onClick={() => setOrdenDir((d) => (d === 'desc' ? 'asc' : 'desc'))}
                    title="Cambiar el sentido del orden"
                    className={`${claseSelect} inline-flex items-center gap-1.5 whitespace-nowrap hover:border-blue-500 hover:text-white`}
                  >
                    {ordenDir === 'desc' ? <ArrowDownWideNarrow className="w-4 h-4" /> : <ArrowUpNarrowWide className="w-4 h-4" />}
                    {ordenCampo === 'fecha'
                      ? (ordenDir === 'desc' ? 'Más recientes' : 'Más antiguos')
                      : (ordenDir === 'desc' ? 'Mayor a menor' : 'Menor a mayor')}
                  </button>
                </div>
              </div>

              <div className="grid grid-cols-1 sm:grid-cols-2 lg:grid-cols-4 gap-3">
                <label className="flex flex-col gap-1 text-xs text-slate-400">
                  Categoría
                  <select
                    value={filtroCategoria}
                    onChange={(e) => { setFiltroCategoria(e.target.value); setFiltroSubcategoria(''); }}
                    className={claseSelect}
                  >
                    <option value="">Todas</option>
                    {arbolCategorias.map((g) => (
                      <option key={g.categoria} value={g.categoria}>{g.categoria}</option>
                    ))}
                  </select>
                </label>
                <label className="flex flex-col gap-1 text-xs text-slate-400">
                  Subcategoría
                  <select
                    value={filtroSubcategoria}
                    onChange={(e) => setFiltroSubcategoria(e.target.value)}
                    disabled={!filtroCategoria}
                    className={`${claseSelect} disabled:opacity-40`}
                  >
                    <option value="">Todas</option>
                    {subcategoriasDisponibles.map((sub) => (
                      <option key={sub} value={sub}>{sub}</option>
                    ))}
                  </select>
                </label>
                <label className="flex flex-col gap-1 text-xs text-slate-400">
                  Mes
                  <select
                    value={filtroMes}
                    onChange={(e) => { setFiltroMes(e.target.value); setFiltroDia(''); }}
                    className={`${claseSelect} capitalize`}
                  >
                    <option value="">Todos</option>
                    {mesesDisponibles.map((m) => (
                      <option key={m} value={m}>{nombreMes(m)}</option>
                    ))}
                  </select>
                </label>
                <label className="flex flex-col gap-1 text-xs text-slate-400">
                  Día
                  <input
                    type="date"
                    value={filtroDia}
                    onChange={(e) => { setFiltroDia(e.target.value); if (e.target.value) setFiltroMes(''); }}
                    className={`${claseSelect} [color-scheme:dark]`}
                  />
                </label>
              </div>

              <div className="flex flex-wrap items-center justify-between gap-3 text-sm">
                <p className="text-slate-400">
                  <span className="text-white font-semibold">{ticketsFiltrados.length}</span>{' '}
                  {ticketsFiltrados.length === 1 ? 'ticket' : 'tickets'} ·{' '}
                  {filtroCategoria ? `gasto en ${filtroSubcategoria || filtroCategoria}` : 'gasto total'}:{' '}
                  <span className="font-mono text-emerald-400">{gastoFiltrado.toFixed(2)} €</span>
                </p>
                {hayFiltros && (
                  <button onClick={limpiarFiltros} className="inline-flex items-center gap-1 text-slate-400 hover:text-white">
                    <X className="w-4 h-4" /> Limpiar filtros
                  </button>
                )}
              </div>
            </div>

            {loadingDatos ? (
              <div className="text-center py-10 text-slate-500">Cargando datos...</div>
            ) : ticketsFiltrados.length === 0 ? (
              <div className="text-center py-10 text-slate-500 bg-slate-900 rounded-xl border border-slate-800">
                {tickets.length === 0 ? 'No hay tickets procesados todavía.'
                  : busqueda ? `Ningún ticket se llama «${busqueda}» con los filtros actuales.` : 'Ningún ticket coincide con los filtros.'}
              </div>
            ) : (
              <div className="space-y-6">
                {ticketsFiltrados.map((ticket, index) => {
                  const subtotal = ticket._items.reduce((acc, it) => acc + importeItem(it), 0);
                  return (
                    <div key={ticket.id || index} className="bg-slate-900 border border-slate-800 rounded-xl p-6">
                      <div className="flex justify-between items-start mb-4 border-b border-slate-800 pb-4">
                        <div>
                          <h3 className="text-lg font-bold text-white">
                            {ticket.nombre_personalizado || ticket.tienda}
                          </h3>
                          <p className="text-sm text-slate-400">{formatoFecha(ticket._fecha) || ticket.fecha_compra}</p>
                        </div>
                        <div className="flex items-start gap-4">
                        <div className="text-right">
                          {filtroCategoria ? (
                            <>
                              <p className="text-sm text-slate-400">En {filtroSubcategoria || filtroCategoria}</p>
                              <p className="text-2xl font-mono text-emerald-400">{subtotal.toFixed(2)} €</p>
                              <p className="text-xs text-slate-500 mt-1">
                                {ticket._items.length} de {ticket.items?.length || 0} productos · total ticket {Number(ticket.precio_total).toFixed(2)} €
                              </p>
                            </>
                          ) : (
                            <>
                              <p className="text-sm text-slate-400">Total</p>
                              <p className="text-2xl font-mono text-emerald-400">{Number(ticket.precio_total).toFixed(2)} €</p>
                            </>
                          )}
                        </div>
                        {/* Eliminar ticket: primer clic pide confirmación, segundo clic borra */}
                        {confirmarBorrado === ticket.id ? (
                          <div className="flex flex-col items-end gap-1">
                            <span className="text-xs text-slate-400">¿Eliminar?</span>
                            <div className="flex gap-1">
                              <button
                                onClick={() => handleDeleteTicket(ticket)}
                                disabled={borrando === ticket.id}
                                className="px-3 py-1.5 rounded-lg bg-red-600 hover:bg-red-500 text-white text-xs font-medium disabled:opacity-50 inline-flex items-center gap-1"
                              >
                                {borrando === ticket.id ? <Loader2 className="w-3.5 h-3.5 animate-spin" /> : <Trash2 className="w-3.5 h-3.5" />}
                                Sí
                              </button>
                              <button
                                onClick={() => setConfirmarBorrado(null)}
                                disabled={borrando === ticket.id}
                                className="px-3 py-1.5 rounded-lg border border-slate-700 text-slate-300 hover:bg-slate-800 text-xs"
                              >
                                No
                              </button>
                            </div>
                          </div>
                        ) : (
                          <button
                            onClick={() => setConfirmarBorrado(ticket.id)}
                            title="Eliminar ticket"
                            aria-label="Eliminar ticket"
                            className="p-2.5 rounded-lg bg-red-600 hover:bg-red-500 text-white transition-colors"
                          >
                            <Trash2 className="w-5 h-5" />
                          </button>
                        )}
                        </div>
                      </div>

                      <table className="w-full text-sm text-left">
                        <thead className="text-xs text-slate-500 uppercase bg-slate-950/50">
                          <tr>
                            <th className="py-2 px-3">Producto</th>
                            <th className="py-2 px-3 text-center">Cant.</th>
                            <th className="py-2 px-3 text-right">Precio/U</th>
                            <th className="py-2 px-3">Categorías</th>
                          </tr>
                        </thead>
                        <tbody>
                          {ticket._items.map((item, i) => (
                            <tr key={i} className="border-b border-slate-800/50">
                              <td className="py-2 px-3 font-medium text-slate-300">{item.descripcion}</td>
                              <td className="py-2 px-3 text-center text-slate-400">{item.cantidad}</td>
                              <td className="py-2 px-3 text-right font-mono text-slate-300">{Number(item.precio_unitario).toFixed(2)} €</td>
                              <td className="py-2 px-3">
                                <div className="flex flex-wrap gap-1">
                                  {(item.categorias || []).map((cat) => (
                                    <span
                                      key={`${cat.categoria}-${cat.subcategoria}`}
                                      title={cat.categoria}
                                      className={`rounded-full px-2 py-0.5 text-xs ${
                                        cat.categoria === filtroCategoria
                                          ? 'bg-blue-500/20 text-blue-300 border border-blue-500/40'
                                          : 'bg-slate-800 text-slate-300'
                                      }`}
                                    >
                                      {cat.subcategoria || cat.categoria}
                                    </span>
                                  ))}
                                </div>
                              </td>
                            </tr>
                          ))}
                        </tbody>
                      </table>
                    </div>
                  );
                })}
              </div>
            )}
          </section>
        )}

      </div>
    </div>
  );
}