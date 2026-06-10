import { useEffect, useRef, useCallback } from 'react';
import { message } from 'antd';

interface WsEvent {
  type: string;
  payload: Record<string, unknown>;
  timestamp: string;
}

const eventHandlers = new Map<string, Set<(event: WsEvent) => void>>();

export function subscribeWsEvent(type: string, handler: (event: WsEvent) => void) {
  if (!eventHandlers.has(type)) {
    eventHandlers.set(type, new Set());
  }
  eventHandlers.get(type)!.add(handler);
  return () => {
    eventHandlers.get(type)?.delete(handler);
  };
}

function emitWsEvent(event: WsEvent) {
  eventHandlers.get(event.type)?.forEach((handler) => {
    try {
      handler(event);
    } catch (e) {
      console.error('WS handler error:', e);
    }
  });
}

let wsInstance: WebSocket | null = null;
let reconnectTimer: ReturnType<typeof setTimeout> | null = null;
let reconnectAttempts = 0;
const MAX_RECONNECT = 10;

function connectWs() {
  if (wsInstance?.readyState === WebSocket.OPEN || wsInstance?.readyState === WebSocket.CONNECTING) {
    return;
  }

  const protocol = window.location.protocol === 'https:' ? 'wss:' : 'ws:';
  const wsUrl = `${protocol}//${window.location.host}/ws/milvus`;

  try {
    wsInstance = new WebSocket(wsUrl);

    wsInstance.onopen = () => {
      console.log('WS connected');
      reconnectAttempts = 0;
      if (reconnectTimer) clearTimeout(reconnectTimer);
    };

    wsInstance.onmessage = (event) => {
      try {
        const data = JSON.parse(event.data) as WsEvent;
        emitWsEvent(data);
      } catch (e) {
        console.error('WS parse error:', e);
      }
    };

    wsInstance.onclose = () => {
      console.log('WS disconnected');
      wsInstance = null;
      // 自动重连
      if (reconnectAttempts < MAX_RECONNECT) {
        const delay = Math.min(1000 * Math.pow(2, reconnectAttempts), 30000);
        reconnectTimer = setTimeout(() => {
          reconnectAttempts++;
          connectWs();
        }, delay);
      }
    };

    wsInstance.onerror = (e) => {
      console.error('WS error:', e);
    };
  } catch (e) {
    console.error('WS init error:', e);
  }
}

export function useWebSocket() {
  const connectedRef = useRef(false);

  useEffect(() => {
    if (!connectedRef.current) {
      connectedRef.current = true;
      connectWs();
    }
    return () => {
      // Don't disconnect on unmount — the singleton manages itself
    };
  }, []);

  const subscribe = useCallback((type: string, handler: (event: WsEvent) => void) => {
    return subscribeWsEvent(type, handler);
  }, []);

  return { subscribe };
}
