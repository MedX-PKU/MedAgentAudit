import type { InjectionKey, Ref } from 'vue'

export type AnnotationDrawerContext = {
  isOpen: Ref<boolean>
  toggle: () => void
  close: () => void
}

export const ANNOTATION_DRAWER_KEY: InjectionKey<AnnotationDrawerContext> = Symbol('annotationDrawer')

