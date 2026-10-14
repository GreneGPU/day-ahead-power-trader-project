/* PostHog browser analytics. Configuration comes from Vercel runtime settings. */
(() => {
  // Your own browsers: open any page once with ?internal=1 to mark this browser as internal (?internal=0 undoes
  // it). Its events are still sent, tagged internal_user=true, so PostHog's "Filter out internal and test users"
  // setting (internal_user "is not set") can leave them out. The mark lives in this browser's storage only.
  const INTERNAL_KEY = 'power-trader-internal';
  let internal = false;
  try {
    const url = new URL(window.location.href), flag = url.searchParams.get('internal');
    if (flag === '1') localStorage.setItem(INTERNAL_KEY, '1');
    if (flag === '0') localStorage.removeItem(INTERNAL_KEY);
    internal = localStorage.getItem(INTERNAL_KEY) === '1';
    if (flag === '1' || flag === '0') {
      url.searchParams.delete('internal');
      history.replaceState(history.state, '', url.pathname + url.search + url.hash);
      const note = document.createElement('div');
      note.setAttribute('role', 'status');
      note.textContent = flag === '1' ? 'This browser is marked as yours: its visits are tagged internal in analytics.'
        : 'This browser is no longer marked as internal.';
      note.style.cssText = 'position:fixed;left:16px;bottom:16px;z-index:9999;max-width:calc(100% - 32px);padding:10px 14px;'
        + 'border-radius:8px;background:#0a2540;color:#fff;font:14px/1.4 system-ui,sans-serif;box-shadow:0 8px 24px rgba(0,0,0,.25)';
      const show = () => { document.body.append(note); setTimeout(() => note.remove(), 5000); };
      if (document.body) show(); else document.addEventListener('DOMContentLoaded', show, {once: true});
    }
  } catch {
    // Storage can be blocked; the visit is then simply not marked.
  }

  if (['localhost', '127.0.0.1', '[::1]'].includes(window.location.hostname)) return;

  async function startAnalytics() {
    try {
      const response = await fetch('/api/analytics-config', {
        cache: 'no-store',
        signal: AbortSignal.timeout(10000),
      });
      if (!response.ok) return;
      const config = await response.json();
      if (!config.enabled || !config.token?.startsWith('phc_') ||
          !['https://eu.i.posthog.com', 'https://us.i.posthog.com'].includes(config.host)) return;

      // Official async snippet: https://posthog.com/docs/libraries/js
      !function(t,e){var o,n,p,r;e.__SV||(window.posthog=e,e._i=[],e.init=function(i,s,a){function g(t,e){var o=e.split('.');2==o.length&&(t=t[o[0]],e=o[1]),t[e]=function(){t.push([e].concat(Array.prototype.slice.call(arguments,0)))}}(p=t.createElement('script')).type='text/javascript',p.crossOrigin='anonymous',p.async=!0,p.src=s.api_host.replace('.i.posthog.com','-assets.i.posthog.com')+'/static/array.js',(r=t.getElementsByTagName('script')[0]).parentNode.insertBefore(p,r);var u=e;for(void 0!==a?u=e[a]=[]:a='posthog',u.people=u.people||[],Object.defineProperty(u,'toString',{configurable:!0,enumerable:!0,writable:!0,value:function(t){var e='posthog';return'posthog'!==a&&(e+='.'+a),t||(e+=' (stub)'),e}}),Object.defineProperty(u.people,'toString',{configurable:!0,enumerable:!0,writable:!0,value:function(){return u.toString(1)+'.people (stub)'}}),o='init capture register register_once register_for_session unregister unregister_for_session getFeatureFlag getFeatureFlagResult isFeatureEnabled reloadFeatureFlags updateEarlyAccessFeatureEnrollment getEarlyAccessFeatures on onFeatureFlags onSessionId getSurveys getActiveMatchingSurveys renderSurvey canRenderSurvey getNextSurveyStep identify setPersonProperties group resetGroups setPersonPropertiesForFlags resetPersonPropertiesForFlags setGroupPropertiesForFlags resetGroupPropertiesForFlags reset get_distinct_id getGroups get_session_id get_session_replay_url alias set_config startSessionRecording stopSessionRecording sessionRecordingStarted captureException loadToolbar get_property getSessionProperty createPersonProfile opt_in_capturing opt_out_capturing has_opted_in_capturing has_opted_out_capturing clear_opt_in_out_capturing debug'.split(' '),n=0;n<o.length;n++)g(u,o[n]);e._i.push([i,s,a])},e.__SV=1)}(document,window.posthog||[]);

      window.posthog.init(config.token, {
        api_host: config.host,
        defaults: '2026-05-30',
        capture_pageview: true,
        capture_pageleave: true,
        person_profiles: 'identified_only',
        autocapture: true,
        disable_session_recording: false,
        session_recording: {
          maskAllInputs: true,
        },
        enable_recording_console_log: false,
        capture_dead_clicks: true,
        capture_heatmaps: true,
        capture_exceptions: false,
        capture_performance: false,
        disable_surveys: true,
        // Tags every event, including the first pageview and replay data, from a browser marked internal.
        before_send: event => {
          if (internal && event) event.properties = {...event.properties, internal_user: true};
          return event;
        },
      });
    } catch {
      // Analytics outages or blocking must never interrupt the dashboard.
    }
  }

  startAnalytics();
})();
