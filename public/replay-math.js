(function(root) {
  'use strict';
  const api = {
    strength(signal, settings) {
      if (signal == null || !Number.isFinite(signal)) return null;
      const half = (settings.upper - settings.lower) / 2;
      if (!(half > 0)) return null;
      const centered = (signal - (settings.upper + settings.lower) / 2) / half;
      return centered * (settings.direction === 'buy_high' ? 1 : -1) * 100;
    },
    stats(rows, capital) {
      let pnl = 0, peak = 0, drawdown = 0, fees = 0, active = 0, warmup = 0;
      for (const row of rows) {
        pnl += row.Cashflow;
        peak = Math.max(peak, pnl);
        drawdown = Math.max(drawdown, peak - pnl);
        fees += row.Transaction_Cost || 0;
        // Prop runs have a position; battery runs are active while charging or discharging.
        active += Number(row.Position !== undefined ? row.Position !== 0 : Boolean(row.Dispatch_MW));
        warmup += Number(row.Custom_Signal == null);
      }
      return {total_cashflow:pnl,max_drawdown:drawdown,total_fee_cost:fees,
        active_intervals:active,warmup_intervals:warmup,ending_equity_dkk:capital+pnl};
    }
  };
  if (typeof module !== 'undefined' && module.exports) module.exports = api;
  else root.ReplayMath = api;
})(typeof window !== 'undefined' ? window : globalThis);
