from typing import Optional
import datetime
import decimal
import uuid

from sqlalchemy import Boolean, DateTime, ForeignKeyConstraint, Integer, Numeric, PrimaryKeyConstraint, String, Text, UniqueConstraint, Uuid, text
from sqlalchemy.dialects.postgresql import JSONB
from sqlalchemy.orm import DeclarativeBase, Mapped, mapped_column, relationship

class Base(DeclarativeBase):
    pass


class Templates(Base):
    __tablename__ = 'templates'
    __table_args__ = (
        PrimaryKeyConstraint('id', name='templates_pkey'),
    )

    id: Mapped[uuid.UUID] = mapped_column(Uuid, primary_key=True, server_default=text('gen_random_uuid()'))
    name: Mapped[str] = mapped_column(Text, nullable=False)
    category: Mapped[str] = mapped_column(String(30), nullable=False)
    created_at: Mapped[Optional[datetime.datetime]] = mapped_column(DateTime(True), server_default=text('CURRENT_TIMESTAMP'))
    updated_at: Mapped[Optional[datetime.datetime]] = mapped_column(DateTime(True), server_default=text('CURRENT_TIMESTAMP'))

    template_versions: Mapped[list['TemplateVersions']] = relationship('TemplateVersions', back_populates='template')


class Users(Base):
    __tablename__ = 'users'
    __table_args__ = (
        PrimaryKeyConstraint('id', name='users_pkey'),
        UniqueConstraint('email', name='users_email_key')
    )

    id: Mapped[uuid.UUID] = mapped_column(Uuid, primary_key=True, server_default=text('gen_random_uuid()'))
    name: Mapped[str] = mapped_column(String(100), nullable=False)
    email: Mapped[str] = mapped_column(String(100), nullable=False)
    role: Mapped[str] = mapped_column(String(100), nullable=False)
    password_hash: Mapped[str] = mapped_column(Text, nullable=False)
    created_at: Mapped[Optional[datetime.datetime]] = mapped_column(DateTime(True), server_default=text('CURRENT_TIMESTAMP'))
    updated_at: Mapped[Optional[datetime.datetime]] = mapped_column(DateTime(True), server_default=text('CURRENT_TIMESTAMP'))

    knowledge_base: Mapped[list['KnowledgeBase']] = relationship('KnowledgeBase', back_populates='user')
    chat_sessions: Mapped[list['ChatSessions']] = relationship('ChatSessions', back_populates='user')
    analyses: Mapped[list['Analyses']] = relationship('Analyses', back_populates='user')
    document_versions: Mapped[list['DocumentVersions']] = relationship('DocumentVersions', back_populates='users')


class KnowledgeBase(Base):
    __tablename__ = 'knowledge_base'
    __table_args__ = (
        ForeignKeyConstraint(['user_id'], ['users.id'], name='knowledge_base_user_id_fkey'),
        PrimaryKeyConstraint('id', name='knowledge_base_pkey')
    )

    id: Mapped[uuid.UUID] = mapped_column(Uuid, primary_key=True, server_default=text('gen_random_uuid()'))
    user_id: Mapped[uuid.UUID] = mapped_column(Uuid, nullable=False)
    name: Mapped[str] = mapped_column(String(100), nullable=False)
    description: Mapped[str] = mapped_column(Text, nullable=False)
    is_active: Mapped[Optional[bool]] = mapped_column(Boolean, server_default=text('true'))
    created_at: Mapped[Optional[datetime.datetime]] = mapped_column(DateTime(True), server_default=text('CURRENT_TIMESTAMP'))
    updated_at: Mapped[Optional[datetime.datetime]] = mapped_column(DateTime(True), server_default=text('CURRENT_TIMESTAMP'))

    user: Mapped['Users'] = relationship('Users', back_populates='knowledge_base')
    chat_sessions: Mapped[list['ChatSessions']] = relationship('ChatSessions', back_populates='knowledge_base')
    kb_documents: Mapped[list['KbDocuments']] = relationship('KbDocuments', back_populates='knowledge_base')


class TemplateVersions(Base):
    __tablename__ = 'template_versions'
    __table_args__ = (
        ForeignKeyConstraint(['template_id'], ['templates.id'], name='template_versions_template_id_fkey'),
        PrimaryKeyConstraint('id', name='template_versions_pkey')
    )

    id: Mapped[uuid.UUID] = mapped_column(Uuid, primary_key=True, server_default=text('gen_random_uuid()'))
    template_id: Mapped[uuid.UUID] = mapped_column(Uuid, nullable=False)
    version_no: Mapped[int] = mapped_column(Integer, nullable=False)
    content: Mapped[str] = mapped_column(Text, nullable=False)
    created_at: Mapped[Optional[datetime.datetime]] = mapped_column(DateTime(True), server_default=text('CURRENT_TIMESTAMP'))

    template: Mapped['Templates'] = relationship('Templates', back_populates='template_versions')


class ChatSessions(Base):
    __tablename__ = 'chat_sessions'
    __table_args__ = (
        ForeignKeyConstraint(['knowledge_base_id'], ['knowledge_base.id'], name='chat_sessions_knowledge_base_id_fkey'),
        ForeignKeyConstraint(['user_id'], ['users.id'], name='chat_sessions_user_id_fkey'),
        PrimaryKeyConstraint('id', name='chat_sessions_pkey')
    )

    id: Mapped[uuid.UUID] = mapped_column(Uuid, primary_key=True, server_default=text('gen_random_uuid()'))
    user_id: Mapped[uuid.UUID] = mapped_column(Uuid, nullable=False)
    knowledge_base_id: Mapped[uuid.UUID] = mapped_column(Uuid, nullable=False)
    title: Mapped[str] = mapped_column(Text, nullable=False)
    created_at: Mapped[Optional[datetime.datetime]] = mapped_column(DateTime(True), server_default=text('CURRENT_TIMESTAMP'))
    updated_at: Mapped[Optional[datetime.datetime]] = mapped_column(DateTime(True), server_default=text('CURRENT_TIMESTAMP'))

    knowledge_base: Mapped['KnowledgeBase'] = relationship('KnowledgeBase', back_populates='chat_sessions')
    user: Mapped['Users'] = relationship('Users', back_populates='chat_sessions')
    analyses: Mapped[list['Analyses']] = relationship('Analyses', back_populates='session')
    chat_messages: Mapped[list['ChatMessages']] = relationship('ChatMessages', back_populates='session')


class KbDocuments(Base):
    __tablename__ = 'kb_documents'
    __table_args__ = (
        ForeignKeyConstraint(['knowledge_base_id'], ['knowledge_base.id'], name='kb_documents_knowledge_base_id_fkey'),
        ForeignKeyConstraint(['user_id'], ['users.id'], name='kb_documents_user_id_fkey'),
        PrimaryKeyConstraint('id', name='kb_documents_pkey')
    )

    id: Mapped[uuid.UUID] = mapped_column(Uuid, primary_key=True, server_default=text('gen_random_uuid()'))
    knowledge_base_id: Mapped[uuid.UUID] = mapped_column(Uuid, nullable=False)
    user_id: Mapped[uuid.UUID] = mapped_column(Uuid, nullable=True)
    title: Mapped[str] = mapped_column(Text, nullable=False)
    document_type: Mapped[str] = mapped_column(String(50), nullable=False)
    status: Mapped[str] = mapped_column(String(20), nullable=False)
    created_at: Mapped[Optional[datetime.datetime]] = mapped_column(DateTime(True), server_default=text('CURRENT_TIMESTAMP'))
    updated_at: Mapped[Optional[datetime.datetime]] = mapped_column(DateTime(True), server_default=text('CURRENT_TIMESTAMP'))

    knowledge_base: Mapped['KnowledgeBase'] = relationship('KnowledgeBase', back_populates='kb_documents')
    analyses: Mapped[list['Analyses']] = relationship('Analyses', back_populates='document')
    document_versions: Mapped[list['DocumentVersions']] = relationship('DocumentVersions', back_populates='kb_document')


class Analyses(Base):
    __tablename__ = 'analyses'
    __table_args__ = (
        ForeignKeyConstraint(['document_id'], ['kb_documents.id'], name='analyses_document_id_fkey'),
        ForeignKeyConstraint(['session_id'], ['chat_sessions.id'], name='analyses_session_id_fkey'),
        ForeignKeyConstraint(['user_id'], ['users.id'], name='analyses_user_id_fkey'),
        PrimaryKeyConstraint('id', name='analyses_pkey')
    )

    id: Mapped[uuid.UUID] = mapped_column(Uuid, primary_key=True, server_default=text('gen_random_uuid()'))
    user_id: Mapped[uuid.UUID] = mapped_column(Uuid, nullable=False)
    document_id: Mapped[uuid.UUID] = mapped_column(Uuid, nullable=False)
    session_id: Mapped[uuid.UUID] = mapped_column(Uuid, nullable=False)
    query: Mapped[str] = mapped_column(Text, nullable=False)
    response: Mapped[str] = mapped_column(Text, nullable=False)
    confidence_score: Mapped[decimal.Decimal] = mapped_column(Numeric, nullable=False)
    sources_used: Mapped[dict] = mapped_column(JSONB, nullable=False)
    created_at: Mapped[Optional[datetime.datetime]] = mapped_column(DateTime(True), server_default=text('CURRENT_TIMESTAMP'))

    document: Mapped['KbDocuments'] = relationship('KbDocuments', back_populates='analyses')
    session: Mapped['ChatSessions'] = relationship('ChatSessions', back_populates='analyses')
    user: Mapped['Users'] = relationship('Users', back_populates='analyses')


class ChatMessages(Base):
    __tablename__ = 'chat_messages'
    __table_args__ = (
        ForeignKeyConstraint(['session_id'], ['chat_sessions.id'], name='chat_messages_session_id_fkey'),
        PrimaryKeyConstraint('id', name='chat_messages_pkey')
    )

    id: Mapped[uuid.UUID] = mapped_column(Uuid, primary_key=True, server_default=text('gen_random_uuid()'))
    session_id: Mapped[uuid.UUID] = mapped_column(Uuid, nullable=False)
    role: Mapped[str] = mapped_column(String(20), nullable=False)
    message: Mapped[str] = mapped_column(Text, nullable=False)
    tokens_used: Mapped[int] = mapped_column(Integer, nullable=False)
    created_at: Mapped[Optional[datetime.datetime]] = mapped_column(DateTime(True), server_default=text('CURRENT_TIMESTAMP'))

    session: Mapped['ChatSessions'] = relationship('ChatSessions', back_populates='chat_messages')


class DocumentVersions(Base):
    __tablename__ = 'document_versions'
    __table_args__ = (
        ForeignKeyConstraint(['created_by'], ['users.id'], name='document_versions_created_by_fkey'),
        ForeignKeyConstraint(['kb_document_id'], ['kb_documents.id'], name='document_versions_kb_document_id_fkey'),
        PrimaryKeyConstraint('id', name='document_versions_pkey')
    )

    id: Mapped[uuid.UUID] = mapped_column(Uuid, primary_key=True, server_default=text('gen_random_uuid()'))
    kb_document_id: Mapped[uuid.UUID] = mapped_column(Uuid, nullable=False)
    version_no: Mapped[int] = mapped_column(Integer, nullable=False)
    content: Mapped[str] = mapped_column(Text, nullable=False)
    embedding_status: Mapped[str] = mapped_column(String(20), nullable=False)
    created_by: Mapped[uuid.UUID] = mapped_column(Uuid, nullable=False)
    created_at: Mapped[Optional[datetime.datetime]] = mapped_column(DateTime(True), server_default=text('CURRENT_TIMESTAMP'))

    users: Mapped['Users'] = relationship('Users', back_populates='document_versions')
    kb_document: Mapped['KbDocuments'] = relationship('KbDocuments', back_populates='document_versions')
