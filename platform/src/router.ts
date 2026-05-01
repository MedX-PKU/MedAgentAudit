import { createRouter, createWebHistory } from 'vue-router'

import AnnotationLayout from './views/annotation/AnnotationLayout.vue'
import AuditView from './views/annotation/audit/AuditView.vue'
import HomeView from './views/home/HomeView.vue'
import OpenCodingView from './views/annotation/open-coding/OpenCodingView.vue'

const router = createRouter({
  history: createWebHistory(import.meta.env.BASE_URL),
  routes: [
    { path: '/', name: 'home', component: HomeView },
    {
      path: '/annotation',
      component: AnnotationLayout,
      children: [
        { path: '', redirect: { name: 'open-coding' } },
        { path: 'open-coding', name: 'open-coding', component: OpenCodingView },
        { path: 'audit', name: 'audit', component: AuditView },
      ],
    },
  ],
})

export default router
