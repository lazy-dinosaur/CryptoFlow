/**
 * CryptoFlow Analytics Tracker
 * Lightweight, privacy-friendly analytics
 */
(function () {
    'use strict';

    // Respect Do Not Track
    if (navigator.doNotTrack === '1') return;

    // Respect Cookie Consent
    const consent = localStorage.getItem('cf_cookie_consent');
    if (consent === 'declined') return;

    // Generate session ID (persists for this browser session)
    const sessionId = sessionStorage.getItem('cf_session') ||
        Math.random().toString(36).substring(2) + Date.now().toString(36);
    sessionStorage.setItem('cf_session', sessionId);

    // Track pageview on load
    function trackPageview() {
        fetch('/api/analytics/pageview', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({
                path: window.location.pathname,
                referrer: document.referrer || null,
                sessionId: sessionId
            })
        }).catch(() => { }); // Silently fail
    }

    // Track clicks
    function trackClick(e) {
        const target = e.target;
        if (!target) return;

        // Only track meaningful clicks (buttons, links, elements with IDs)
        const isInteractive = target.tagName === 'BUTTON' ||
            target.tagName === 'A' ||
            target.id ||
            target.closest('button, a, [data-track]');

        if (!isInteractive) return;

        const element = target.closest('button, a, [data-track]') || target;

        fetch('/api/analytics/click', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({
                path: window.location.pathname,
                elementId: element.id || null,
                elementClass: element.className?.split?.(' ')?.[0] || null,
                elementTag: element.tagName,
                x: Math.round(e.clientX),
                y: Math.round(e.clientY),
                sessionId: sessionId
            })
        }).catch(() => { }); // Silently fail
    }

    // Initialize
    if (document.readyState === 'loading') {
        document.addEventListener('DOMContentLoaded', trackPageview);
    } else {
        trackPageview();
    }

    document.addEventListener('click', trackClick, { passive: true });

    // Expose for debugging
    window.CryptoFlowAnalytics = { sessionId };
})();
